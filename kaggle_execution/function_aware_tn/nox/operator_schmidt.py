"""Projection-aware two-site MPO construction for NEC.

The four-site NEC core constrains three Schmidt cuts simultaneously.  This
module replaces that core with one screened operator-Schmidt cut at the same
coefficient budget.  A rank-r operator-Schmidt expansion is a two-site MPO,
or equivalently a sum of r Kronecker products.  It can be applied without
materialising the dense weight matrix.

The sparse shield and activation bridge remain part of the representation,
but their decoded contributions and the tensor core are jointly rescaled by a
small closed-form solve under a normalized weight-plus-activation objective.
There is no gradient optimization or post-finalization update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nox.utils import quantize_dequantize_groupwise


CODEC_TYPE = "nec_operator_schmidt_v1"


@dataclass(frozen=True)
class OperatorSchmidtLayout:
    name: str
    output_factors: tuple[int, int]
    input_factors: tuple[int, int]
    partition_axes: tuple[int, int, int, int]

    @property
    def dimensions(self) -> tuple[int, int, int, int]:
        return (*self.output_factors, *self.input_factors)

    @property
    def unfolding_shape(self) -> tuple[int, int]:
        dims = self.dimensions
        axes = self.partition_axes
        return dims[axes[0]] * dims[axes[1]], dims[axes[2]] * dims[axes[3]]


def _factor_pair(value: int) -> tuple[int, int]:
    """Return the closest integer factor pair, larger factor first."""
    root = math.isqrt(int(value))
    for divisor in range(root, 0, -1):
        if value % divisor == 0:
            return value // divisor, divisor
    raise RuntimeError("positive integers always have a factor pair")


def candidate_layouts(
    d_out: int,
    d_in: int,
    projection_name: str | None = None,
    *,
    n_heads: int = 32,
    head_dim: int = 128,
) -> tuple[OperatorSchmidtLayout, ...]:
    """Build exact coordinate layouts; no layout changes the dense operator.

    Attention candidates explicitly test both the intuitive head/head cut and
    a crossed cut.  The latter retains head coordinates but balances the two
    Schmidt dimensions.  MLP candidates align the 11,008 neuron axis with
    43x256 groups and also test a more balanced exact factorization.
    """
    layouts: list[OperatorSchmidtLayout] = []

    def add(name, output_factors, input_factors, axes):
        if math.prod(output_factors) != d_out or math.prod(input_factors) != d_in:
            return
        item = OperatorSchmidtLayout(name, tuple(output_factors), tuple(input_factors), tuple(axes))
        signature = (item.output_factors, item.input_factors, item.partition_axes)
        if all((x.output_factors, x.input_factors, x.partition_axes) != signature for x in layouts):
            layouts.append(item)

    is_attention = projection_name in {"q_proj", "k_proj", "v_proj", "o_proj"}
    if is_attention and d_out == n_heads * head_dim and d_in == n_heads * head_dim:
        add("attention_head_head", (n_heads, head_dim), (n_heads, head_dim), (0, 2, 1, 3))
        add("attention_head_crossed", (n_heads, head_dim), (n_heads, head_dim), (0, 3, 1, 2))

    if d_out == 11008 and d_in == 4096:
        add("swiglu_neuron_group", (43, 256), (32, 128), (0, 2, 1, 3))
        add("swiglu_neuron_crossed", (43, 256), (32, 128), (0, 3, 1, 2))
        add("swiglu_balanced_crossed", (86, 128), (64, 64), (0, 3, 1, 2))
    elif d_out == 4096 and d_in == 11008:
        add("swiglu_neuron_group", (32, 128), (43, 256), (0, 2, 1, 3))
        add("swiglu_neuron_crossed", (32, 128), (43, 256), (0, 3, 1, 2))
        add("swiglu_balanced_crossed", (64, 64), (86, 128), (0, 3, 1, 2))

    out_pair = _factor_pair(d_out)
    in_pair = _factor_pair(d_in)
    add("balanced_direct", out_pair, in_pair, (0, 2, 1, 3))
    add("balanced_crossed", out_pair, in_pair, (0, 3, 1, 2))
    return tuple(layouts)


def four_site_parameter_count(o_modes, i_modes, ranks) -> int:
    if len(o_modes) != 4 or len(i_modes) != 4 or len(ranks) != 5:
        raise ValueError("expected four physical sites and five TT ranks")
    return int(sum(
        ranks[index] * o_modes[index] * i_modes[index] * ranks[index + 1]
        for index in range(4)
    ))


def _matrix_to_unfolding(matrix: torch.Tensor, layout: OperatorSchmidtLayout) -> torch.Tensor:
    axes = layout.partition_axes
    dims = layout.dimensions
    rows, columns = layout.unfolding_shape
    return matrix.reshape(*dims).permute(*axes).reshape(rows, columns).contiguous()


def _unfolding_to_matrix(unfolding: torch.Tensor, layout: OperatorSchmidtLayout) -> torch.Tensor:
    dims = layout.dimensions
    axes = layout.partition_axes
    permuted_dims = tuple(dims[index] for index in axes)
    inverse = tuple(axes.index(index) for index in range(4))
    return unfolding.reshape(*permuted_dims).permute(*inverse).reshape(
        math.prod(layout.output_factors), math.prod(layout.input_factors)
    )


def _truncated_svd(
    matrix: torch.Tensor,
    rank: int,
    *,
    oversample: int,
    power_iterations: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic exact or randomized SVD with bounded auxiliary storage."""
    rows, columns = matrix.shape
    rank = min(int(rank), rows, columns)
    if rank <= 0:
        raise ValueError("operator-Schmidt rank must be positive")
    if matrix.numel() <= 2_000_000 or rank == min(rows, columns):
        left, values, right = torch.linalg.svd(matrix.float(), full_matrices=False)
        return left[:, :rank], values[:rank], right[:rank]

    sample_rank = min(rank + int(oversample), rows, columns)
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(int(seed))
    # Sketch the smaller orientation.  This matters for the 1024x16384 head
    # layout and avoids constructing a needlessly large orthogonal basis.
    transposed = rows > columns
    source = matrix.T if transposed else matrix
    omega = torch.randn(
        source.shape[1], sample_rank, device=source.device, dtype=source.dtype,
        generator=generator,
    )
    sample = source @ omega
    for _ in range(int(power_iterations)):
        sample = source @ (source.T @ sample)
    basis = torch.linalg.qr(sample, mode="reduced").Q
    small = basis.T @ source
    u_small, values, right = torch.linalg.svd(small, full_matrices=False)
    left = basis @ u_small
    left, values, right = left[:, :rank], values[:rank], right[:rank]
    if transposed:
        return right.T.contiguous(), values, left.T.contiguous()
    return left.contiguous(), values, right.contiguous()


def operator_schmidt_to_dense(left: torch.Tensor, right: torch.Tensor, layout) -> torch.Tensor:
    layout = _layout_from_value(layout)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("factors must have shapes [rank, rows] and [rank, columns]")
    unfolding = left.T @ right
    return _unfolding_to_matrix(unfolding, layout)


def _layout_from_value(value) -> OperatorSchmidtLayout:
    if isinstance(value, OperatorSchmidtLayout):
        return value
    return OperatorSchmidtLayout(
        value["name"], tuple(value["output_factors"]), tuple(value["input_factors"]),
        tuple(value["partition_axes"]),
    )


def apply_operator_schmidt_factors(
    left: torch.Tensor,
    right: torch.Tensor,
    layout,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Apply a two-site MPO without reconstructing its dense matrix."""
    layout = _layout_from_value(layout)
    og, ol = layout.output_factors
    ig, il = layout.input_factors
    if inputs.ndim != 2 or inputs.shape[0] != ig * il:
        raise ValueError(f"expected inputs [{ig * il}, N], got {tuple(inputs.shape)}")
    x = inputs.reshape(ig, il, inputs.shape[1])
    rank = left.shape[0]
    if layout.partition_axes == (0, 2, 1, 3):
        a = left.reshape(rank, og, ig)
        b = right.reshape(rank, ol, il)
        output = torch.einsum("kac,kbd,cdn->abn", a, b, x)
    elif layout.partition_axes == (0, 3, 1, 2):
        a = left.reshape(rank, og, il)
        b = right.reshape(rank, ol, ig)
        output = torch.einsum("kad,kbc,cdn->abn", a, b, x)
    else:
        raise ValueError(f"unsupported two-site partition: {layout.partition_axes}")
    return output.reshape(og * ol, inputs.shape[1])


def _add_sparse_output(output, indices, values, inputs, chunk_size=65536):
    if indices is None or values is None:
        return output
    indices = indices.to(device=inputs.device, dtype=torch.long)
    values = values.to(device=inputs.device, dtype=inputs.dtype)
    for start in range(0, values.numel(), chunk_size):
        part = slice(start, min(start + chunk_size, values.numel()))
        rows = indices[part, 0]
        columns = indices[part, 1]
        output.index_add_(0, rows, values[part, None] * inputs.index_select(0, columns))
    return output


def apply_operator_schmidt_artifact(artifact, inputs, *, dtype=None):
    """Execute an NEC operator-Schmidt artifact directly in factorized form."""
    if artifact.get("codec_type") != CODEC_TYPE:
        raise ValueError(f"unsupported artifact type: {artifact.get('codec_type')!r}")
    target_dtype = dtype or inputs.dtype
    x = inputs.to(dtype=target_dtype)
    scale = artifact["input_scale"].to(device=x.device, dtype=target_dtype)
    left = artifact["left_factor"].to(device=x.device, dtype=target_dtype)
    right = artifact["right_factor"].to(device=x.device, dtype=target_dtype)
    output = apply_operator_schmidt_factors(left, right, artifact["layout"], x / scale[:, None])
    if artifact.get("A_bridge") is not None:
        a = artifact["A_bridge"].to(device=x.device, dtype=target_dtype)
        b = artifact["B_bridge"].to(device=x.device, dtype=target_dtype)
        output.add_(a @ (b @ x))
    return _add_sparse_output(
        output, artifact.get("sparse_indices"), artifact.get("sparse_values"), x,
    )


def reconstruct_operator_schmidt(artifact, *, device="cpu", dtype=torch.float32):
    rows, columns = tuple(artifact["orig_shape"])
    left = artifact["left_factor"].to(device=device, dtype=dtype)
    right = artifact["right_factor"].to(device=device, dtype=dtype)
    scale = artifact["input_scale"].to(device=device, dtype=dtype)
    result = operator_schmidt_to_dense(left, right, artifact["layout"]) / scale[None, :]
    if artifact.get("A_bridge") is not None:
        a = artifact["A_bridge"].to(device=device, dtype=dtype)
        b = artifact["B_bridge"].to(device=device, dtype=dtype)
        result.add_(a @ b)
    if artifact.get("sparse_indices") is not None:
        indices = artifact["sparse_indices"].to(device=device, dtype=torch.long)
        values = artifact["sparse_values"].to(device=device, dtype=dtype)
        result[indices[:, 0], indices[:, 1]] += values
    if tuple(result.shape) != (rows, columns):
        raise RuntimeError("decoded operator shape does not match artifact metadata")
    return result


def _normalized_objective(weight, target_output, candidate, inputs, weight_mix, activation_mix):
    weight_term = (weight - candidate).square().sum() / weight.square().sum().clamp_min(1e-30)
    activation_term = (target_output - candidate @ inputs).square().sum()
    activation_term = activation_term / target_output.square().sum().clamp_min(1e-30)
    return weight_mix * weight_term + activation_mix * activation_term


def _joint_component_gains(
    weight,
    target_output,
    inputs,
    components,
    *,
    weight_mix,
    activation_mix,
    ridge=1e-6,
):
    """Jointly solve the component amplitudes under the exact mixed loss."""
    active = [component for component in components if component is not None]
    if not active:
        return [], {"condition_number": None, "ridge": 0.0}
    outputs = [component @ inputs for component in active]
    weight_energy = weight.square().sum().clamp_min(1e-30)
    output_energy = target_output.square().sum().clamp_min(1e-30)
    count = len(active)
    gram = torch.empty(count, count, device=weight.device, dtype=torch.float64)
    rhs = torch.empty(count, device=weight.device, dtype=torch.float64)
    for i in range(count):
        rhs[i] = (
            weight_mix * (active[i] * weight).sum() / weight_energy
            + activation_mix * (outputs[i] * target_output).sum() / output_energy
        ).double()
        for j in range(i, count):
            value = (
                weight_mix * (active[i] * active[j]).sum() / weight_energy
                + activation_mix * (outputs[i] * outputs[j]).sum() / output_energy
            ).double()
            gram[i, j] = gram[j, i] = value
    ridge_value = float(ridge) * float(gram.diagonal().mean().clamp_min(1e-30))
    regularized = gram + ridge_value * torch.eye(count, device=gram.device, dtype=gram.dtype)
    gains = torch.linalg.solve(
        regularized, rhs + ridge_value * torch.ones_like(rhs)
    ).to(dtype=torch.float32).clamp(-4.0, 4.0)
    condition = torch.linalg.cond(regularized)
    return [float(value) for value in gains], {
        "condition_number": float(condition), "ridge": ridge_value,
    }


def estimate_operator_schmidt_bytes(artifact) -> int:
    keys = (
        "left_factor", "right_factor", "input_scale", "A_bridge", "B_bridge",
        "sparse_indices", "sparse_values",
    )
    return int(sum(
        artifact[key].numel() * artifact[key].element_size()
        for key in keys if artifact.get(key) is not None
    ))


@torch.no_grad()
def nox_operator_schmidt_decompose(
    W,
    X_calib,
    Y_dense,
    o_modes=(8, 8, 8, 8),
    i_modes=(8, 8, 8, 8),
    ranks=(1, 24, 64, 24, 1),
    r_bridge=64,
    n_outliers=262144,
    return_artifacts=False,
    verbose=True,
    quantize_components=True,
    projection_name=None,
    n_heads=32,
    weight_mix=1.0,
    activation_mix=1.0,
    oversample=8,
    power_iterations=1,
    seed=0,
    **_ignored,
):
    """Construct a screened two-site NEC core at the legacy MPO core budget."""
    if W.ndim != 2 or X_calib.ndim != 2 or Y_dense.ndim != 2:
        raise ValueError("weight, calibration inputs, and targets must be matrices")
    if W.shape[1] != X_calib.shape[0] or W.shape[0] != Y_dense.shape[0]:
        raise ValueError("incompatible weight/input/target dimensions")
    if X_calib.shape[1] != Y_dense.shape[1]:
        raise ValueError("input and target token counts differ")
    if weight_mix < 0 or activation_mix < 0 or weight_mix + activation_mix <= 0:
        raise ValueError("mixed-objective weights must be nonnegative and not both zero")
    if n_outliers < 0 or r_bridge < 0:
        raise ValueError("component budgets must be nonnegative")
    device = W.device
    source = W.float()
    inputs = X_calib.float()
    targets = Y_dense.float()
    if quantize_components:
        quantize = lambda value: quantize_dequantize_groupwise(value, bits=4, group_size=64)
    else:
        quantize = lambda value: value

    channel_rms = inputs.square().mean(dim=1).add(1e-8).sqrt()
    if n_outliers:
        count = min(int(n_outliers), source.numel())
        importance = source.abs() * channel_rms[None, :]
        selected = torch.topk(importance.reshape(-1), count, sorted=False).indices
        mask = torch.zeros(source.numel(), device=device, dtype=torch.bool)
        mask[selected] = True
        mask = mask.reshape_as(source)
        sparse = quantize(torch.where(mask, source, torch.zeros_like(source))) * mask
    else:
        mask = torch.zeros_like(source, dtype=torch.bool)
        sparse = torch.zeros_like(source)

    bridge_left = bridge_right = None
    bridge = torch.zeros_like(source)
    if r_bridge:
        u_x, s_x, vh_x = torch.linalg.svd(inputs, full_matrices=False)
        activation_rank = min(s_x.numel(), max(2 * int(r_bridge), 64))
        residual_output = targets - sparse @ inputs
        projected = residual_output @ vh_x[:activation_rank].T
        u_y, _, _ = torch.linalg.svd(projected, full_matrices=False)
        actual_rank = min(int(r_bridge), u_y.shape[1])
        bridge_left = u_y[:, :actual_rank]
        denominator = (
            s_x[:activation_rank].square() + (0.02 * s_x[0]).square()
        ).clamp_min(1e-30)
        regularized_inverse = s_x[:activation_rank] / denominator
        bridge_right = (
            (bridge_left.T @ projected) * regularized_inverse[None, :]
        ) @ u_x[:, :activation_rank].T
        norms = torch.sqrt(
            bridge_left.norm(dim=0) / bridge_right.norm(dim=1).clamp_min(1e-8)
        )
        bridge_left = quantize(bridge_left / norms[None, :])
        bridge_right = quantize(bridge_right * norms[:, None])
        bridge = bridge_left @ bridge_right

    residual = source - sparse - bridge
    weight_energy = source.square().sum().clamp_min(1e-30)
    output_energy = targets.square().sum().clamp_min(1e-30)
    diagonal_metric = (
        weight_mix / weight_energy
        + activation_mix * inputs.square().sum(dim=1) / output_energy
    ).clamp_min(torch.finfo(torch.float32).tiny)
    # The scale is stored as FP16. Keep it representable there even for a
    # degenerate activation-only objective (e.g. an all-zero calibration set).
    input_scale = diagonal_metric.sqrt().clamp_min(torch.finfo(torch.float16).tiny)
    conditioned = residual * input_scale[None, :]
    budget = four_site_parameter_count(o_modes, i_modes, ranks)

    candidate_records = []
    best = None
    for layout_index, layout in enumerate(candidate_layouts(
        *source.shape, projection_name, n_heads=n_heads,
        head_dim=(source.shape[0] // n_heads if source.shape[0] == 4096 else 128),
    )):
        rows, columns = layout.unfolding_shape
        rank = min(rows, columns, budget // (rows + columns))
        if rank <= 0:
            continue
        unfolding = _matrix_to_unfolding(conditioned, layout)
        u, values, vh = _truncated_svd(
            unfolding, rank, oversample=oversample,
            power_iterations=power_iterations, seed=seed + layout_index,
        )
        root = values.clamp_min(0).sqrt()
        left = quantize((u * root[None, :]).T.contiguous())
        right = quantize((root[:, None] * vh).contiguous())
        core = operator_schmidt_to_dense(left, right, layout) / input_scale[None, :]
        candidate = sparse + bridge + core
        objective = float(_normalized_objective(
            source, targets, candidate, inputs, weight_mix, activation_mix,
        ))
        retained = float(values.square().sum() / conditioned.square().sum().clamp_min(1e-30))
        record = {
            "name": layout.name,
            "output_factors": layout.output_factors,
            "input_factors": layout.input_factors,
            "partition_axes": layout.partition_axes,
            "unfolding_shape": (rows, columns),
            "rank": rank,
            "coefficient_count": rank * (rows + columns),
            "conditioned_spectral_energy_retained": retained,
            "mixed_objective_before_joint_rescale": objective,
        }
        candidate_records.append(record)
        if best is None or objective < best[0]:
            best = (objective, layout, left, right, core, record)
    if best is None:
        raise RuntimeError("no operator-Schmidt layout fits the core coefficient budget")
    _, layout, left, right, core, selected_record = best

    gains, gain_diagnostics = _joint_component_gains(
        source, targets, inputs, (sparse, bridge if r_bridge else None, core),
        weight_mix=weight_mix, activation_mix=activation_mix,
    )
    gain_index = 0
    sparse_gain = gains[gain_index]
    gain_index += 1
    bridge_gain = gains[gain_index] if r_bridge else 0.0
    gain_index += int(bool(r_bridge))
    core_gain = gains[gain_index]
    sparse.mul_(sparse_gain)
    if r_bridge:
        bridge_left.mul_(bridge_gain)
        bridge.mul_(bridge_gain)
    left.mul_(core_gain)
    core.mul_(core_gain)
    reconstructed = sparse + bridge + core
    final_objective = float(_normalized_objective(
        source, targets, reconstructed, inputs, weight_mix, activation_mix,
    ))

    sparse_indices = torch.nonzero(mask & (sparse != 0), as_tuple=False).to(
        dtype=torch.int32, device="cpu"
    )
    sparse_values = sparse[mask & (sparse != 0)].to(dtype=torch.float16, device="cpu")
    layout_dict = {
        "name": layout.name,
        "output_factors": layout.output_factors,
        "input_factors": layout.input_factors,
        "partition_axes": layout.partition_axes,
    }
    artifact = {
        "codec_type": CODEC_TYPE,
        "quantization_bits": 4 if quantize_components else 16,
        "left_factor": left.to(dtype=torch.float16, device="cpu"),
        "right_factor": right.to(dtype=torch.float16, device="cpu"),
        "input_scale": input_scale.to(dtype=torch.float16, device="cpu"),
        "A_bridge": (
            bridge_left.to(dtype=torch.float16, device="cpu") if bridge_left is not None else None
        ),
        "B_bridge": (
            bridge_right.to(dtype=torch.float16, device="cpu") if bridge_right is not None else None
        ),
        "sparse_indices": sparse_indices,
        "sparse_values": sparse_values,
        "orig_shape": tuple(source.shape),
        "layout": layout_dict,
        "rank": int(left.shape[0]),
        "legacy_four_site_core_budget": budget,
        "core_coefficient_count": int(left.numel() + right.numel()),
        "component_gains": {
            "sparse": sparse_gain, "bridge": bridge_gain, "operator_schmidt": core_gain,
        },
        "joint_gain_diagnostics": gain_diagnostics,
        "objective": {
            "kind": "normalized_weight_plus_activation",
            "weight_mix": float(weight_mix),
            "activation_mix": float(activation_mix),
            "value": final_objective,
        },
        "layout_candidates": candidate_records,
        "selected_layout": selected_record,
        "gradient_healing": False,
    }
    artifact["payload_bytes"] = estimate_operator_schmidt_bytes(artifact)
    artifact["total_kb"] = artifact["payload_bytes"] / 1024.0
    if verbose:
        print(
            f"    [Nox OS] {projection_name or 'projection'} selected {layout.name} "
            f"rank={left.shape[0]} core={left.numel() + right.numel():,}/{budget:,} "
            f"objective={final_objective:.6f}",
            flush=True,
        )
    factors = [left, right]
    chosen_ranks = [1, int(left.shape[0]), 1]
    if return_artifacts:
        return reconstructed, artifact["total_kb"], factors, chosen_ranks, artifact
    return reconstructed, artifact["total_kb"], factors, chosen_ranks
