"""Training-free low-rank SwiGLU constructions and matched-budget baselines.

The function-aware candidate uses activation-weighted SVD factors for Gate
and Up, then solves a ridge-regularized low-rank output-readout problem for
Down against dense MLP outputs.  For fixed compressed Gate/Up features, the
Down step is a closed-form truncated SVD in the feature covariance metric.
The complete module uses only the stored factors at inference; no dense
residual or teacher is part of the artifact.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def low_rank_coefficient_count(d_out: int, d_in: int, rank: int):
    if min(int(d_out), int(d_in), int(rank)) <= 0:
        raise ValueError("matrix dimensions and rank must be positive")
    return int(rank) * (int(d_out) + int(d_in))


def swiglu_dense_coefficient_count(d_model: int, intermediate_size: int):
    return 3 * int(d_model) * int(intermediate_size)


def swiglu_factor_coefficient_count(
    d_model: int,
    intermediate_size: int,
    ranks: tuple[int, int, int],
):
    gate_rank, up_rank, down_rank = map(int, ranks)
    return (
        low_rank_coefficient_count(intermediate_size, d_model, gate_rank)
        + low_rank_coefficient_count(intermediate_size, d_model, up_rank)
        + low_rank_coefficient_count(d_model, intermediate_size, down_rank)
    )


@torch.no_grad()
def _truncated_svd_factors(
    matrix: torch.Tensor,
    rank: int,
    *,
    randomized_threshold: int = 2_000_000,
    oversample: int = 24,
    power_iterations: int = 2,
    seed: int = 0,
):
    """Return A,B with AB the deterministic rank-r truncated-SVD estimate."""
    source = matrix.detach().float()
    if source.ndim != 2 or not torch.isfinite(source).all():
        raise ValueError("matrix must be finite and rank two")
    rows, columns = source.shape
    rank = min(int(rank), rows, columns)
    if rank <= 0:
        raise ValueError("rank must be positive")
    if source.numel() <= randomized_threshold or rank == min(rows, columns):
        u, singular, vh = torch.linalg.svd(source, full_matrices=False)
        u, singular, vh = u[:, :rank], singular[:rank], vh[:rank]
    else:
        sketch_rank = min(rank + int(oversample), rows, columns)
        generator = torch.Generator(device=source.device)
        generator.manual_seed(int(seed))
        transpose = rows < columns
        work = source.T if transpose else source
        omega = torch.randn(
            work.shape[1], sketch_rank,
            device=source.device, dtype=source.dtype, generator=generator,
        )
        basis = torch.linalg.qr(work @ omega, mode="reduced").Q
        for _ in range(max(0, int(power_iterations))):
            basis = torch.linalg.qr(work @ (work.T @ basis), mode="reduced").Q
        small = basis.T @ work
        u_small, singular, vh_small = torch.linalg.svd(small, full_matrices=False)
        left = basis @ u_small[:, :rank]
        singular = singular[:rank]
        right = vh_small[:rank]
        if transpose:
            u, vh = right.T.contiguous(), left.T.contiguous()
        else:
            u, vh = left.contiguous(), right.contiguous()
    return u * singular.unsqueeze(0), vh


@torch.no_grad()
def ordinary_svd_factors(weight: torch.Tensor, rank: int, *, seed: int = 0):
    return _truncated_svd_factors(weight, rank, seed=seed)


@torch.no_grad()
def activation_weighted_svd_factors(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    rank: int,
    *,
    ridge_ratio: float = 0.01,
    seed: int = 0,
):
    """Best rank-r factorization for ``||(W-W_hat) X||`` with ridge whitening.

    ``inputs`` is [input_dim, samples].  The regularized channel covariance is
    ``XXᵀ/N + ridge*I``.  The truncation is an exact weighted SVD when the
    internal SVD is exact; large randomized cases report the seed separately.
    """
    if inputs.ndim != 2 or weight.ndim != 2 or inputs.shape[0] != weight.shape[1]:
        raise ValueError("weight and activation dimensions are incompatible")
    if inputs.shape[1] <= 0 or ridge_ratio < 0:
        raise ValueError("inputs must be nonempty and ridge_ratio nonnegative")
    x = inputs.detach().float()
    covariance = (x @ x.T) / x.shape[1]
    return activation_weighted_svd_factors_from_covariance(
        weight, covariance, rank, ridge_ratio=ridge_ratio, seed=seed,
    )


@torch.no_grad()
def activation_weighted_svd_factors_from_covariance(
    weight: torch.Tensor,
    covariance: torch.Tensor,
    rank: int,
    *,
    ridge_ratio: float = 0.01,
    seed: int = 0,
):
    """Activation-weighted factorization from a streamed input covariance."""
    if weight.ndim != 2 or covariance.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("weight and covariance dimensions are incompatible")
    if not torch.isfinite(covariance).all() or ridge_ratio < 0:
        raise ValueError("covariance must be finite and ridge_ratio nonnegative")
    chol, _ridge = regularized_covariance_cholesky(covariance, ridge_ratio)
    return activation_weighted_svd_factors_from_cholesky(weight, chol, rank, seed=seed)


def regularized_covariance_cholesky(covariance: torch.Tensor, ridge_ratio: float = 0.01):
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    if not torch.isfinite(covariance).all() or ridge_ratio < 0:
        raise ValueError("covariance must be finite and ridge_ratio nonnegative")
    covariance = covariance.detach().float()
    scale = torch.trace(covariance) / max(1, covariance.shape[0])
    ridge = float(ridge_ratio) * scale.clamp_min(1e-12)
    covariance = covariance + ridge * torch.eye(
        covariance.shape[0], device=covariance.device, dtype=covariance.dtype,
    )
    chol = torch.linalg.cholesky(covariance)
    return chol, ridge


@torch.no_grad()
def activation_weighted_svd_factors_from_cholesky(
    weight: torch.Tensor, chol: torch.Tensor, rank: int, *, seed: int = 0,
):
    if weight.ndim != 2 or chol.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("weight and covariance Cholesky dimensions are incompatible")
    w = weight.detach().float()
    weighted = w @ chol
    left_scaled, right_weighted = _truncated_svd_factors(weighted, rank, seed=seed)
    right = torch.linalg.solve_triangular(
        chol.T, right_weighted.T.contiguous(), upper=True,
    ).T
    return left_scaled, right


@torch.no_grad()
def fit_functional_down_readout(
    features: torch.Tensor,
    targets: torch.Tensor,
    rank: int,
    *,
    ridge_ratio: float = 0.01,
    seed: int = 0,
):
    """Fit a rank-r Down operator to module outputs for fixed features.

    The objective is ``mean(||Y-DH||²) + λ||D||²``.  If ``H/√N=U S Vᵀ``,
    the rank-r solution reduces to truncated SVD of
    ``(Y/√N)V diag(S/sqrt(S²+λ))``; the resulting right factor is mapped back
    through ``Uᵀ``.  This is a closed-form module-output fit, not gradient
    training.
    """
    if features.ndim != 2 or targets.ndim != 2 or features.shape[1] != targets.shape[1]:
        raise ValueError("features and targets must share their sample columns")
    if features.shape[1] <= 0 or ridge_ratio < 0:
        raise ValueError("features must be nonempty and ridge_ratio nonnegative")
    h = features.detach().float()
    y = targets.detach().float()
    normalized = h / math.sqrt(h.shape[1])
    requested_rank = min(min(normalized.shape), int(rank) + 64)
    if normalized.numel() > 2_000_000 and requested_rank < min(normalized.shape):
        feature_left_scaled, vh_h = _truncated_svd_factors(
            normalized, requested_rank, seed=seed + 17,
        )
        singular_h = torch.linalg.vector_norm(feature_left_scaled, dim=0)
        u_h = feature_left_scaled / singular_h.clamp_min(1e-12).unsqueeze(0)
        feature_svd_method = "randomized_range_svd"
    else:
        u_h, singular_h, vh_h = torch.linalg.svd(normalized, full_matrices=False)
        feature_svd_method = "exact_svd"
    if singular_h.numel() == 0:
        raise ValueError("features have zero numerical rank")
    ridge = float(ridge_ratio) * (singular_h.square().mean().clamp_min(1e-12))
    right_targets = (y / math.sqrt(y.shape[1])) @ vh_h.T
    whitening = singular_h / torch.sqrt(singular_h.square() + ridge)
    weighted_target = right_targets * whitening.unsqueeze(0)
    down_left, weighted_right = _truncated_svd_factors(weighted_target, rank, seed=seed)
    down_right = (weighted_right / torch.sqrt(singular_h.square() + ridge).unsqueeze(0)) @ u_h.T
    return down_left, down_right, {
        "objective": "mean_module_output_squared_error_plus_l2_down_prior",
        "ridge_ratio": float(ridge_ratio),
        "ridge_value": float(ridge),
        "feature_numerical_rank": int(torch.count_nonzero(
            singular_h > singular_h.max() * max(h.shape) * torch.finfo(singular_h.dtype).eps,
        )),
        "feature_svd_method": feature_svd_method,
        "feature_svd_rank": int(singular_h.numel()),
        "feature_svd_captured_energy_fraction": float(
            singular_h.square().sum() / normalized.square().sum().clamp_min(1e-30)
        ),
        "down_rank": int(min(rank, down_left.shape[1])),
        "uses_gradients": False,
        "uses_validation": False,
    }


def apply_factorized_swiglu(artifact: dict, inputs: torch.Tensor):
    required = ("gate_left", "gate_right", "up_left", "up_right", "down_left", "down_right")
    if any(name not in artifact for name in required):
        raise KeyError("factorized SwiGLU artifact is missing a matrix factor")
    gate = artifact["gate_left"] @ (artifact["gate_right"] @ inputs)
    up = artifact["up_left"] @ (artifact["up_right"] @ inputs)
    hidden = F.silu(gate) * up
    return artifact["down_left"] @ (artifact["down_right"] @ hidden)


@torch.no_grad()
def build_factorized_swiglu(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    fit_inputs: torch.Tensor,
    ranks: tuple[int, int, int],
    *,
    construction: str = "function_aware",
    ridge_ratio: float = 0.01,
    storage_dtype: torch.dtype | None = None,
    seed: int = 0,
    sketch_size: int | None = None,
    sketch_chunk_size: int = 512,
    fit_input_cholesky: torch.Tensor | None = None,
):
    """Construct a no-residual factorized SwiGLU under a shared fit split.

    ``construction='function_aware'`` uses activation-weighted Gate/Up SVDs
    and fits Down to dense module outputs.  ``construction='svd'`` uses
    ordinary weight SVD for Gate/Up before the same functional Down fit; this
    serves as the explicitly MoDeGPT-inspired module-output control.  The
    latter is not a reproduction of MoDeGPT's full Transformer procedure.
    """
    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("Gate and Up must have matching rank-two shapes")
    intermediate_size, d_model = gate_weight.shape
    if down_weight.shape != (d_model, intermediate_size):
        raise ValueError("Down must have shape [d_model, intermediate_size]")
    if fit_inputs.ndim != 2 or fit_inputs.shape[0] != d_model:
        raise ValueError("fit_inputs must have shape [d_model, samples]")
    ranks = tuple(map(int, ranks))
    if len(ranks) != 3 or any(rank <= 0 for rank in ranks):
        raise ValueError("ranks must contain positive Gate, Up, and Down ranks")
    if construction not in {"function_aware", "svd"}:
        raise ValueError("construction must be 'function_aware' or 'svd'")

    x = fit_inputs.detach().float()
    gate = gate_weight.detach().float()
    up = up_weight.detach().float()
    down = down_weight.detach().float()
    if construction == "function_aware":
        if fit_input_cholesky is None:
            gate_left, gate_right = activation_weighted_svd_factors(
                gate, x, ranks[0], ridge_ratio=ridge_ratio, seed=seed,
            )
            up_left, up_right = activation_weighted_svd_factors(
                up, x, ranks[1], ridge_ratio=ridge_ratio, seed=seed + 1,
            )
        else:
            gate_left, gate_right = activation_weighted_svd_factors_from_cholesky(
                gate, fit_input_cholesky, ranks[0], seed=seed,
            )
            up_left, up_right = activation_weighted_svd_factors_from_cholesky(
                up, fit_input_cholesky, ranks[1], seed=seed + 1,
            )
    else:
        gate_left, gate_right = ordinary_svd_factors(gate, ranks[0], seed=seed)
        up_left, up_right = ordinary_svd_factors(up, ranks[1], seed=seed + 1)

    if sketch_size is not None and construction == "function_aware":
        sketch_size = min(int(sketch_size), x.shape[1])
        if sketch_size <= 0 or sketch_chunk_size <= 0:
            raise ValueError("sketch_size and sketch_chunk_size must be positive")
        generator = torch.Generator(device=x.device)
        generator.manual_seed(int(seed) + 409)
        hidden_sketch = torch.zeros(
            intermediate_size, sketch_size, device=x.device, dtype=torch.float32,
        )
        target_sketch = torch.zeros(
            d_model, sketch_size, device=x.device, dtype=torch.float32,
        )
        for start in range(0, x.shape[1], int(sketch_chunk_size)):
            stop = min(x.shape[1], start + int(sketch_chunk_size))
            x_chunk = x[:, start:stop]
            dense_hidden = F.silu(gate @ x_chunk) * (up @ x_chunk)
            dense_target = down @ dense_hidden
            approximate_hidden = F.silu(gate_left @ (gate_right @ x_chunk)) * (
                up_left @ (up_right @ x_chunk)
            )
            omega = torch.randn(
                stop - start, sketch_size, generator=generator,
                device=x.device, dtype=torch.float32,
            ) / math.sqrt(x.shape[1])
            hidden_sketch.add_(approximate_hidden @ omega)
            target_sketch.add_(dense_target @ omega)
        readout_features, readout_targets = hidden_sketch, target_sketch
        sketch_metadata = {
            "method": "shared_gaussian_token_sketch_of_joint_feature_and_target_covariances",
            "sketch_size": int(sketch_size),
            "chunk_size": int(sketch_chunk_size),
            "total_fit_tokens": int(x.shape[1]),
            "seed": int(seed) + 409,
            "all_fit_tokens_streamed": True,
        }
    else:
        dense_targets = down @ (F.silu(gate @ x) * (up @ x))
        approximate_hidden = F.silu(gate_left @ (gate_right @ x)) * (
            up_left @ (up_right @ x)
        )
        readout_features, readout_targets = approximate_hidden, dense_targets
        sketch_metadata = {
            "method": "materialized_fit_activations",
            "total_fit_tokens": int(x.shape[1]),
            "all_fit_tokens_used": True,
        }
    down_left, down_right, fit_metadata = fit_functional_down_readout(
        readout_features, readout_targets, ranks[2],
        ridge_ratio=ridge_ratio, seed=seed + 2,
    )
    fit_metadata["fit_statistics"] = sketch_metadata
    target_dtype = storage_dtype or gate_weight.dtype
    artifact = {
        "codec_type": "function_aware_factorized_swiglu_v1",
        "construction": construction,
        "orig_shapes": {
            "gate": tuple(gate_weight.shape),
            "up": tuple(up_weight.shape),
            "down": tuple(down_weight.shape),
        },
        "ranks": ranks,
        "gate_left": gate_left.to(target_dtype),
        "gate_right": gate_right.to(target_dtype),
        "up_left": up_left.to(target_dtype),
        "up_right": up_right.to(target_dtype),
        "down_left": down_left.to(target_dtype),
        "down_right": down_right.to(target_dtype),
        "coefficient_count": swiglu_factor_coefficient_count(
            d_model, intermediate_size, ranks,
        ),
        "fit_samples": int(x.shape[1]),
        "post_compression_optimization": False,
        "dense_residual": False,
        "external_teacher_at_inference": False,
        "down_fit": fit_metadata,
    }
    actual_count = sum(artifact[name].numel() for name in (
        "gate_left", "gate_right", "up_left", "up_right", "down_left", "down_right",
    ))
    if actual_count != artifact["coefficient_count"]:
        raise RuntimeError("factorized SwiGLU coefficient accounting mismatch")
    return artifact


@torch.no_grad()
def build_selected_swiglu(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    fit_inputs: torch.Tensor,
    coefficient_budget: int,
    *,
    ridge_ratio: float = 0.01,
    storage_dtype: torch.dtype | None = None,
    sketch_size: int = 2048,
    chunk_size: int = 512,
):
    """MoDeGPT-inspired shared-neuron selection for SwiGLU.

    Gate and Up retain the same selected output neurons; Down retains the
    matching columns and is refit by ridge regression to dense module outputs.
    Selection uses a diagonal ridge-leverage proxy (per-neuron product
    activation energy), not MoDeGPT's full covariance leverage scores. The
    differences are explicit so this control is not presented as a faithful
    reproduction of the published algorithm.
    """
    if gate_weight.ndim != 2 or up_weight.shape != gate_weight.shape:
        raise ValueError("Gate and Up must have matching rank-two shapes")
    intermediate, d_model = gate_weight.shape
    if down_weight.shape != (d_model, intermediate):
        raise ValueError("Down must have shape [d_model, intermediate_size]")
    if fit_inputs.ndim != 2 or fit_inputs.shape[0] != d_model or fit_inputs.shape[1] == 0:
        raise ValueError("fit_inputs must have shape [d_model, nonzero samples]")
    if ridge_ratio <= 0 or sketch_size <= 0 or chunk_size <= 0:
        raise ValueError("ridge_ratio, sketch_size, and chunk_size must be positive")
    width = min(intermediate, max(1, int(coefficient_budget) // (3 * d_model)))
    x = fit_inputs.detach().float()
    gate = gate_weight.detach().float()
    up = up_weight.detach().float()
    down = down_weight.detach().float()
    score_sums = torch.zeros(intermediate, device=x.device, dtype=torch.float32)
    for start in range(0, x.shape[1], int(chunk_size)):
        x_chunk = x[:, start:start + int(chunk_size)]
        dense_hidden = F.silu(gate @ x_chunk) * (up @ x_chunk)
        score_sums.add_(dense_hidden.square().sum(dim=1))
    scores = score_sums / x.shape[1]
    selected = torch.argsort(scores, descending=True, stable=True)[:width]
    selected_gate = gate.index_select(0, selected)
    selected_up = up.index_select(0, selected)
    sketch_width = min(x.shape[1], int(sketch_size))
    generator = torch.Generator(device=x.device)
    generator.manual_seed(731)
    hidden_sketch = torch.zeros(width, sketch_width, device=x.device, dtype=torch.float32)
    target_sketch = torch.zeros(d_model, sketch_width, device=x.device, dtype=torch.float32)
    for start in range(0, x.shape[1], int(chunk_size)):
        stop = min(x.shape[1], start + int(chunk_size))
        x_chunk = x[:, start:stop]
        dense_hidden = F.silu(gate @ x_chunk) * (up @ x_chunk)
        dense_target = down @ dense_hidden
        selected_hidden = F.silu(selected_gate @ x_chunk) * (selected_up @ x_chunk)
        omega = torch.randn(
            stop - start, sketch_width, generator=generator,
            device=x.device, dtype=torch.float32,
        ) / math.sqrt(x.shape[1])
        hidden_sketch.add_(selected_hidden @ omega)
        target_sketch.add_(dense_target @ omega)

    # The shared Gaussian token sketch estimates the fit covariances while
    # limiting retained calibration activations to O((d_model+m)·sketch_width).
    if width <= sketch_width:
        covariance = hidden_sketch @ hidden_sketch.T / sketch_width
        ridge = ridge_ratio * torch.trace(covariance) / max(1, width)
        covariance.diagonal().add_(ridge)
        cross = target_sketch @ hidden_sketch.T / sketch_width
        down_selected = torch.linalg.solve(covariance, cross.T).T
        solver = "feature_covariance_cholesky_solve"
    else:
        covariance = hidden_sketch.T @ hidden_sketch / sketch_width
        ridge = ridge_ratio * torch.trace(covariance) / max(1, sketch_width)
        covariance.diagonal().add_(ridge)
        solved = torch.linalg.solve(covariance, hidden_sketch.T / sketch_width)
        down_selected = target_sketch @ solved
        solver = "dual_sample_covariance_solve"
    target_dtype = storage_dtype or gate_weight.dtype
    artifact = {
        "codec_type": "modegpt_inspired_shared_neuron_swiglu_v1",
        "construction": "shared_neuron_selection_diagonal_leverage_proxy",
        "gate_proj": selected_gate.to(target_dtype),
        "up_proj": selected_up.to(target_dtype),
        "down_proj": down_selected.to(target_dtype),
        "selected_neurons": selected,
        "selection_scores": scores[selected],
        "coefficient_count": int(3 * d_model * width),
        "neuron_count": int(width),
        "fit_samples": int(x.shape[1]),
        "fit_token_sketch_size": int(sketch_width),
        "fit_token_chunk_size": int(chunk_size),
        "fit_sketch_method": "seeded shared Gaussian sketch of hidden-feature and output cross-covariances",
        "ridge_ratio": float(ridge_ratio),
        "ridge_value": float(ridge),
        "down_fit_solver": solver,
        "selection_method": "top diagonal product-activation ridge-leverage proxy",
        "dense_residual": False,
        "uses_gradients": False,
        "post_compression_optimization": False,
        "external_teacher_at_inference": False,
    }
    if artifact["coefficient_count"] > int(coefficient_budget):
        raise RuntimeError("selected-neuron artifact exceeds requested coefficient budget")
    if sum(artifact[name].numel() for name in ("gate_proj", "up_proj", "down_proj")) != artifact["coefficient_count"]:
        raise RuntimeError("selected-neuron coefficient accounting mismatch")
    return artifact


def apply_selected_swiglu(artifact: dict, inputs: torch.Tensor):
    """Apply the compact shared-neuron SwiGLU baseline."""
    return artifact["down_proj"] @ (
        F.silu(artifact["gate_proj"] @ inputs) * (artifact["up_proj"] @ inputs)
    )


def rank_allocations(total_rank: int, *, offsets: tuple[int, ...] | None = None):
    """Small predeclared allocation set with the same total rank budget."""
    total_rank = int(total_rank)
    if total_rank < 3:
        raise ValueError("total rank budget must be at least three")
    delta = max(1, total_rank // 12) if offsets is None else max(map(abs, offsets), default=0)
    options = {
        "balanced": (total_rank // 3, total_rank // 3, total_rank - 2 * (total_rank // 3)),
        "more_gate": (total_rank // 3 + delta, total_rank // 3 - delta, total_rank - 2 * (total_rank // 3)),
        "more_up": (total_rank // 3 - delta, total_rank // 3 + delta, total_rank - 2 * (total_rank // 3)),
        "more_down": (total_rank // 3 - delta, total_rank // 3, total_rank - 2 * (total_rank // 3) + delta),
    }
    return {name: ranks for name, ranks in options.items() if min(ranks) > 0 and sum(ranks) == total_rank}


def budgeted_swiglu_rank(d_model: int, intermediate_size: int, fraction: float):
    """Largest equal rank for three factors under the requested dense fraction."""
    if not 0 < float(fraction) <= 1:
        raise ValueError("fraction must lie in (0,1]")
    dense = swiglu_dense_coefficient_count(d_model, intermediate_size)
    budget = math.floor(dense * float(fraction))
    per_rank = d_model + intermediate_size
    total_rank = budget // per_rank
    if total_rank < 3:
        raise ValueError("coefficient budget cannot hold rank one for all projections")
    base, remainder = divmod(total_rank, 3)
    ranks = tuple(base + (1 if index < remainder else 0) for index in range(3))
    rank = min(ranks)
    count = total_rank * per_rank
    return {
        "fraction": float(fraction),
        "dense_coefficients": dense,
        "coefficient_budget": budget,
        "equal_rank": rank,
        "ranks": ranks,
        "coefficient_count": count,
        "actual_fraction": count / dense,
        "parameter_reduction": 1.0 - count / dense,
        "parameter_compression_ratio": dense / count,
    }
