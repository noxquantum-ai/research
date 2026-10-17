"""Neuron-coordinate-preserving grouped tensor network for SwiGLU.

The intermediate neuron axis is partitioned into native contiguous groups.
Within each group, Gate, Up, and Down are two-node tensor networks:

    G_q ~= A^G_q B^G_q,  U_q ~= A^U_q B^U_q,
    D_q ~= A^D_q B^D_q.

The coordinatewise nonlinearity is still evaluated at every original neuron.
Because Down is a sum of group terms, its aggregate ordinary matrix rank can
reach min(d_model, groups * rank), avoiding the global shared-rank bottleneck.
"""

from __future__ import annotations

import io

import torch
import torch.nn.functional as F


CODEC_TYPE = "nec_grouped_swiglu_btn_v1"


def grouped_swiglu_coefficient_count(
    d_model: int, intermediate_size: int, groups: int,
    rank: int | tuple[int, int, int],
) -> int:
    d_model = int(d_model)
    intermediate_size = int(intermediate_size)
    groups = int(groups)
    ranks = (int(rank),) * 3 if isinstance(rank, int) else tuple(map(int, rank))
    if len(ranks) != 3:
        raise ValueError("rank must be an integer or (gate, up, down) tuple")
    if min(d_model, intermediate_size, groups, *ranks) <= 0:
        raise ValueError("dimensions, groups, and rank must be positive")
    if intermediate_size % groups:
        raise ValueError("groups must divide intermediate_size")
    group_size = intermediate_size // groups
    if any(value > min(group_size, d_model) for value in ranks):
        raise ValueError("rank exceeds a group matrix dimension")
    return groups * sum(ranks) * (group_size + d_model)


def grouped_rank_for_fraction(
    d_model: int, intermediate_size: int, groups: int, coefficient_fraction: float,
) -> int:
    coefficient_fraction = float(coefficient_fraction)
    if not 0.0 < coefficient_fraction <= 1.0:
        raise ValueError("coefficient_fraction must be in (0, 1]")
    group_size = intermediate_size // groups
    dense = 3 * d_model * intermediate_size
    denominator = 3 * groups * (group_size + d_model)
    rank = int(round(coefficient_fraction * dense / denominator))
    return max(1, min(group_size, d_model, rank))


def _top_eigenvectors(gram: torch.Tensor, rank: int) -> torch.Tensor:
    gram = 0.5 * (gram + gram.T)
    _, vectors = torch.linalg.eigh(gram)
    return vectors[:, -rank:].flip(1).contiguous()


@torch.no_grad()
def _row_projection_factors(weight, inputs, rank, weight_mix, activation_mix):
    weight = weight.float()
    inputs = inputs.float()
    outputs = weight @ inputs
    gram = weight_mix * (weight @ weight.T)
    gram.addmm_(outputs, outputs.T, beta=1.0, alpha=activation_mix / inputs.shape[1])
    left = _top_eigenvectors(gram, rank)
    right = left.T @ weight
    return left, right


@torch.no_grad()
def _column_weighted_factors(weight, inputs, rank, weight_mix, activation_mix):
    """Best rank-r approximation under a weight-plus-input covariance metric."""
    weight = weight.float()
    inputs = inputs.float()
    size = inputs.shape[0]
    covariance = activation_mix * (inputs @ inputs.T) / inputs.shape[1]
    covariance.diagonal().add_(weight_mix)
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    eigenvalues = eigenvalues.clamp_min(1e-8)
    sqrt_cov = (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T
    inv_sqrt_cov = (
        eigenvectors * eigenvalues.rsqrt().unsqueeze(0)
    ) @ eigenvectors.T
    transformed = weight @ sqrt_cov
    small_gram = transformed.T @ transformed
    right_basis = _top_eigenvectors(small_gram, rank)
    left = transformed @ right_basis
    right = right_basis.T @ inv_sqrt_cov
    if left.shape != (weight.shape[0], rank) or right.shape != (rank, size):
        raise RuntimeError("internal grouped Down factor shape mismatch")
    return left, right


@torch.no_grad()
def build_grouped_swiglu_artifact(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    calibration_input: torch.Tensor,
    *,
    groups: int = 32,
    rank: int | None = None,
    rank_gate: int | None = None,
    rank_up: int | None = None,
    rank_down: int | None = None,
    coefficient_fraction: float | None = None,
    weight_mix: float = 1.0,
    activation_mix: float = 1.0,
    storage_dtype: torch.dtype = torch.float16,
) -> dict:
    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("Gate and Up must be same-shaped matrices")
    intermediate_size, d_model = gate_weight.shape
    if down_weight.shape != (d_model, intermediate_size):
        raise ValueError("Down has incompatible shape")
    if calibration_input.ndim != 2 or calibration_input.shape[0] != d_model:
        raise ValueError("calibration_input has incompatible shape")
    if intermediate_size % int(groups):
        raise ValueError("groups must divide intermediate_size")
    explicit_ranks = (rank_gate, rank_up, rank_down)
    if rank is not None and any(value is not None for value in explicit_ranks):
        raise ValueError("provide either rank or an explicit rank profile")
    if rank is not None:
        if coefficient_fraction is not None:
            raise ValueError("rank and coefficient_fraction are mutually exclusive")
        ranks = (int(rank),) * 3
    elif any(value is not None for value in explicit_ranks):
        if coefficient_fraction is not None or any(value is None for value in explicit_ranks):
            raise ValueError("an explicit rank profile requires all three ranks only")
        ranks = tuple(map(int, explicit_ranks))
    elif coefficient_fraction is not None:
        equal_rank = grouped_rank_for_fraction(
            d_model, intermediate_size, int(groups), float(coefficient_fraction),
        )
        ranks = (equal_rank,) * 3
    else:
        raise ValueError("provide rank, a complete rank profile, or coefficient_fraction")
    rank_gate, rank_up, rank_down = ranks
    group_size = intermediate_size // int(groups)
    coefficient_count = grouped_swiglu_coefficient_count(
        d_model, intermediate_size, groups, ranks,
    )
    if weight_mix < 0 or activation_mix < 0 or weight_mix + activation_mix <= 0:
        raise ValueError("objective mixtures must be nonnegative and not both zero")

    x = calibration_input.float()
    gate_left = []
    gate_right = []
    up_left = []
    up_right = []
    down_left = []
    down_right = []
    dense_target = torch.zeros(
        d_model, x.shape[1], device=x.device, dtype=torch.float32,
    )
    latent_features = []
    for group in range(int(groups)):
        start = group * group_size
        stop = start + group_size
        gate_block = gate_weight[start:stop]
        up_block = up_weight[start:stop]
        down_block = down_weight[:, start:stop]
        gl, gr = _row_projection_factors(
            gate_block, x, rank_gate, weight_mix, activation_mix,
        )
        ul, ur = _row_projection_factors(
            up_block, x, rank_up, weight_mix, activation_mix,
        )
        dense_gate = gate_block.float() @ x
        dense_up = up_block.float() @ x
        dense_product = F.silu(dense_gate) * dense_up
        compressed_gate = gl @ (gr @ x)
        compressed_up = ul @ (ur @ x)
        compressed_product = F.silu(compressed_gate) * compressed_up
        dl, dr = _column_weighted_factors(
            down_block, compressed_product, rank_down, weight_mix, activation_mix,
        )
        gate_left.append(gl.to(device="cpu", dtype=storage_dtype))
        gate_right.append(gr.to(device="cpu", dtype=storage_dtype))
        up_left.append(ul.to(device="cpu", dtype=storage_dtype))
        up_right.append(ur.to(device="cpu", dtype=storage_dtype))
        down_left.append(dl.to(device="cpu", dtype=storage_dtype))
        down_right.append(dr.to(device="cpu", dtype=storage_dtype))
        latent_features.append(dr @ compressed_product)
        dense_target.add_(down_block.float() @ dense_product)

    artifact = {
        "codec_type": CODEC_TYPE,
        "orig_shapes": {
            "gate": tuple(gate_weight.shape),
            "up": tuple(up_weight.shape),
            "down": tuple(down_weight.shape),
        },
        "groups": int(groups),
        "group_size": group_size,
        "rank": max(ranks),
        "rank_gate": rank_gate,
        "rank_up": rank_up,
        "rank_down": rank_down,
        "gate_left": torch.stack(gate_left),
        "gate_right": torch.stack(gate_right),
        "up_left": torch.stack(up_left),
        "up_right": torch.stack(up_right),
        "down_left": torch.stack(down_left),
        "down_right": torch.stack(down_right),
        "coefficient_count": coefficient_count,
        "coefficient_fraction_of_dense_mlp": float(
            coefficient_count / (3 * d_model * intermediate_size)
        ),
        "construction_objective": "grouped_weight_plus_activation_operator_projection",
        "weight_mix": float(weight_mix),
        "activation_mix": float(activation_mix),
        "construction_uses_validation": False,
        "post_compression_optimization": False,
        "uses_dense_residual": False,
    }
    _validate_artifact(artifact)
    return artifact, torch.cat(latent_features, dim=0), dense_target


def _validate_artifact(
    artifact: dict,
) -> tuple[int, int, int, int, int, int]:
    if artifact.get("codec_type") != CODEC_TYPE:
        raise ValueError("unsupported grouped SwiGLU artifact")
    groups = int(artifact["groups"])
    group_size = int(artifact["group_size"])
    rank_gate = int(artifact.get("rank_gate", artifact["rank"]))
    rank_up = int(artifact.get("rank_up", artifact["rank"]))
    rank_down = int(artifact.get("rank_down", artifact["rank"]))
    intermediate_size, d_model = tuple(artifact["orig_shapes"]["gate"])
    expected = {
        "gate_left": (groups, group_size, rank_gate),
        "gate_right": (groups, rank_gate, d_model),
        "up_left": (groups, group_size, rank_up),
        "up_right": (groups, rank_up, d_model),
        "down_left": (groups, d_model, rank_down),
        "down_right": (groups, rank_down, group_size),
    }
    for name, shape in expected.items():
        if tuple(artifact[name].shape) != shape:
            raise ValueError(f"invalid {name} shape")
    if groups * group_size != intermediate_size:
        raise ValueError("group metadata does not cover the neuron axis")
    if int(artifact["coefficient_count"]) != grouped_swiglu_coefficient_count(
        d_model, intermediate_size, groups, (rank_gate, rank_up, rank_down),
    ):
        raise ValueError("coefficient accounting mismatch")
    return d_model, intermediate_size, groups, rank_gate, rank_up, rank_down


@torch.no_grad()
def grouped_swiglu_features(artifact: dict, inputs: torch.Tensor) -> torch.Tensor:
    d_model, _, groups, _rank_gate, _rank_up, rank_down = _validate_artifact(artifact)
    if inputs.ndim != 2 or inputs.shape[0] != d_model:
        raise ValueError("inputs have incompatible shape")
    dtype = inputs.dtype
    device = inputs.device
    gl = artifact["gate_left"].to(device=device, dtype=dtype)
    gr = artifact["gate_right"].to(device=device, dtype=dtype)
    ul = artifact["up_left"].to(device=device, dtype=dtype)
    ur = artifact["up_right"].to(device=device, dtype=dtype)
    dr = artifact["down_right"].to(device=device, dtype=dtype)
    gate_latent = torch.einsum("qrd,dn->qrn", gr, inputs)
    up_latent = torch.einsum("qrd,dn->qrn", ur, inputs)
    gate = torch.einsum("qgr,qrn->qgn", gl, gate_latent)
    up = torch.einsum("qgr,qrn->qgn", ul, up_latent)
    product = F.silu(gate) * up
    latent = torch.einsum("qrg,qgn->qrn", dr, product)
    return latent.reshape(groups * rank_down, inputs.shape[1])


@torch.no_grad()
def apply_grouped_swiglu(artifact: dict, inputs: torch.Tensor) -> torch.Tensor:
    d_model, _, groups, _rank_gate, _rank_up, rank_down = _validate_artifact(artifact)
    latent = grouped_swiglu_features(artifact, inputs).reshape(
        groups, rank_down, inputs.shape[1],
    )
    left = artifact["down_left"].to(device=inputs.device, dtype=inputs.dtype)
    output = torch.einsum("qdr,qrn->dn", left, latent)
    if output.shape[0] != d_model:
        raise RuntimeError("grouped SwiGLU output shape mismatch")
    return output


def replace_grouped_down_readout(artifact: dict, readout: torch.Tensor) -> None:
    d_model, _, groups, _rank_gate, _rank_up, rank_down = _validate_artifact(artifact)
    if readout.shape != (d_model, groups * rank_down):
        raise ValueError("readout has incompatible shape")
    artifact["down_left"] = readout.reshape(d_model, groups, rank_down).permute(
        1, 0, 2,
    ).contiguous().to(device="cpu", dtype=artifact["down_left"].dtype)
    _validate_artifact(artifact)


def grouped_swiglu_payload_bytes(artifact: dict) -> int:
    _validate_artifact(artifact)
    return int(sum(
        artifact[name].numel() * artifact[name].element_size()
        for name in (
            "gate_left", "gate_right", "up_left", "up_right",
            "down_left", "down_right",
        )
    ))


def grouped_swiglu_serialized_bytes(artifact: dict) -> int:
    _validate_artifact(artifact)
    stream = io.BytesIO()
    torch.save(artifact, stream)
    return int(stream.tell())


class GroupedSwiGLUModule(torch.nn.Module):
    """Factorized batched runtime for a grouped SwiGLU artifact."""

    def __init__(self, artifact: dict, *, device=None, dtype=None):
        super().__init__()
        _validate_artifact(artifact)
        target_dtype = dtype or artifact["gate_left"].dtype
        for name in (
            "gate_left", "gate_right", "up_left", "up_right",
            "down_left", "down_right",
        ):
            self.register_buffer(
                name, artifact[name].to(device=device, dtype=target_dtype),
                persistent=True,
            )
        self.groups = int(artifact["groups"])
        self.group_size = int(artifact["group_size"])
        self.rank = int(artifact["rank"])
        self.rank_gate = int(artifact.get("rank_gate", artifact["rank"]))
        self.rank_up = int(artifact.get("rank_up", artifact["rank"]))
        self.rank_down = int(artifact.get("rank_down", artifact["rank"]))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_latent = torch.einsum("qrd,...d->...qr", self.gate_right, hidden_states)
        up_latent = torch.einsum("qrd,...d->...qr", self.up_right, hidden_states)
        gate = torch.einsum("qgr,...qr->...qg", self.gate_left, gate_latent)
        up = torch.einsum("qgr,...qr->...qg", self.up_left, up_latent)
        product = F.silu(gate) * up
        down_latent = torch.einsum("qrg,...qg->...qr", self.down_right, product)
        return torch.einsum("qdr,...qr->...d", self.down_left, down_latent)
