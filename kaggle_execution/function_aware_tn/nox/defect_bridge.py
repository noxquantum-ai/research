"""Training-free residual-defect decoders for initial NEC artifact construction.

The bridge predicts the dense-minus-compressed output defect from the actual
compressed activation entering a layer.  It is fitted once, before the final
artifact is written, with a low-rank ridge map plus an optional kernel memory
for nonlinear residual structure.  It is not an optimizer and performs no
updates after artifact serialization.
"""

from __future__ import annotations

import torch


def _as_feature_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 2:
        raise ValueError(f"{name} must be a [features, samples] tensor")
    return value.to(dtype=torch.float32)


def _truncate_factor_product(left: torch.Tensor, right: torch.Tensor, rank: int):
    """Return rank-r factors for left @ right.T using a small-core SVD."""
    q_left, r_left = torch.linalg.qr(left, mode="reduced")
    q_right, r_right = torch.linalg.qr(right, mode="reduced")
    core = r_left @ r_right.T
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    keep = min(int(rank), singular.numel())
    root = singular[:keep].clamp_min(0).sqrt()
    return (
        (q_left @ u[:, :keep]) * root.unsqueeze(0),
        (q_right @ vh[:keep].T) * root.unsqueeze(0),
        singular,
    )

def fit_ridge_low_rank_bridge(
    context: torch.Tensor,
    defect: torch.Tensor,
    *,
    rank: int,
    ridge_ratio: float = 0.01,
):
    """Fit B ~= defect @ pinv(context) with ridge and rank(B) <= rank.

    The normal equations are solved in the calibration-sample subspace. If
    context = U diag(s) V.T, the ridge solution is

        B = defect @ V @ diag(s / (s**2 + lambda)) @ U.T,

    where lambda is scaled by the mean squared singular value. QR factorizations
    and a small-core SVD impose the requested rank without materializing B.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    if ridge_ratio < 0:
        raise ValueError("ridge_ratio must be nonnegative")
    x = _as_feature_matrix(context, "context")
    y = _as_feature_matrix(defect, "defect")
    if x.shape != y.shape:
        raise ValueError("context and defect must have identical shapes")
    if x.shape[1] == 0:
        raise ValueError("at least one calibration sample is required")

    u, singular, vh = torch.linalg.svd(x, full_matrices=False)
    mean_energy = singular.square().mean().clamp_min(torch.finfo(singular.dtype).eps)
    ridge = float(ridge_ratio) * mean_energy
    weights = singular / (singular.square() + ridge)
    left = (y @ vh.T) * weights.unsqueeze(0)
    right = u
    left_factor, right_factor, full_spectrum = _truncate_factor_product(left, right, rank)
    predicted = left_factor @ (right_factor.T @ x)
    metrics = {
        "requested_rank": int(rank),
        "effective_rank": int(left_factor.shape[1]),
        "ridge": float(ridge),
        "calibration_relative_error": float(
            torch.linalg.vector_norm(y - predicted)
            / torch.linalg.vector_norm(y).clamp_min(torch.finfo(y.dtype).eps)
        ),
        "retained_bridge_energy": float(
            full_spectrum[: left_factor.shape[1]].square().sum()
            / full_spectrum.square().sum().clamp_min(torch.finfo(y.dtype).eps)
        ),
    }
    return left_factor, right_factor, metrics


def farthest_point_indices(context: torch.Tensor, count: int) -> torch.Tensor:
    """Deterministically choose activation-space representatives from columns."""
    x = _as_feature_matrix(context, "context")
    count = min(int(count), x.shape[1])
    if count <= 0:
        raise ValueError("count must be positive")
    normalized = torch.nn.functional.normalize(x, dim=0, eps=1e-8)
    # Greedy farthest-point coverage, with deterministic tie-breaking by index.
    selected = torch.empty(count, dtype=torch.long, device=x.device)
    selected[0] = 0
    best_similarity = normalized[:, 0] @ normalized
    for step in range(1, count):
        next_idx = torch.argmin(best_similarity)
        selected[step] = next_idx
        similarity = normalized[:, next_idx] @ normalized
        best_similarity = torch.maximum(best_similarity, similarity)
    return selected


def fit_hybrid_defect_bridge(
    context: torch.Tensor,
    defect: torch.Tensor,
    *,
    linear_rank: int,
    memory_size: int,
    ridge_ratio: float = 0.01,
    memory_top_k: int = 8,
    temperature: float | None = None,
    storage_dtype: torch.dtype = torch.float16,
):
    """Fit a low-rank global correction and a nonlinear residual memory.

    ``context`` and ``defect`` contain calibration examples only. The memory
    stores representative compressed contexts as cosine keys and the residual
    left after the rank-limited linear bridge as values. At inference, a
    top-k cosine kernel interpolates those residual values.
    """
    x = _as_feature_matrix(context, "context")
    y = _as_feature_matrix(defect, "defect")
    if x.shape != y.shape:
        raise ValueError("context and defect must have identical shapes")
    if memory_size < 0 or memory_top_k <= 0:
        raise ValueError("memory_size must be nonnegative and memory_top_k positive")

    left, right, metrics = fit_ridge_low_rank_bridge(
        x, y, rank=linear_rank, ridge_ratio=ridge_ratio,
    )
    residual = y - left @ (right.T @ x)
    memory_count = min(int(memory_size), x.shape[1])
    if memory_count:
        indices = farthest_point_indices(x, memory_count)
        keys = torch.nn.functional.normalize(x[:, indices], dim=0, eps=1e-8)
        values = residual[:, indices]
        if temperature is None:
            if memory_count == 1:
                fitted_temperature = 0.05
            else:
                gram = keys.T @ keys
                gram.fill_diagonal_(-torch.inf)
                nearest = gram.max(dim=1).values
                fitted_temperature = float((1.0 - nearest.median()).clamp(0.01, 0.2))
        else:
            fitted_temperature = float(temperature)
        if fitted_temperature <= 0:
            raise ValueError("temperature must be positive")
        bridge = {
            "linear_left": left.to(dtype=storage_dtype).cpu(),
            "linear_right": right.to(dtype=storage_dtype).cpu(),
            "memory_keys": keys.to(dtype=storage_dtype).cpu(),
            "memory_values": values.to(dtype=storage_dtype).cpu(),
            "memory_top_k": min(int(memory_top_k), memory_count),
            "memory_temperature": fitted_temperature,
            "ridge_ratio": float(ridge_ratio),
            "metadata": {
                **metrics,
                "memory_count": memory_count,
                "memory_top_k": min(int(memory_top_k), memory_count),
                "memory_temperature": fitted_temperature,
                "objective": "ridge_global_plus_topk_cosine_defect_memory",
            },
        }
    else:
        bridge = {
            "linear_left": left.to(dtype=storage_dtype).cpu(),
            "linear_right": right.to(dtype=storage_dtype).cpu(),
            "memory_keys": None,
            "memory_values": None,
            "memory_top_k": 0,
            "memory_temperature": None,
            "ridge_ratio": float(ridge_ratio),
            "metadata": {**metrics, "memory_count": 0, "objective": "ridge_low_rank_defect"},
        }
    bridge["metadata"]["serialized_tensor_bytes"] = sum(
        tensor.numel() * tensor.element_size()
        for tensor in bridge.values()
        if torch.is_tensor(tensor)
    )
    return bridge


def apply_defect_bridge(context: torch.Tensor, bridge: dict) -> torch.Tensor:
    """Evaluate a saved bridge for [features, samples] activations."""
    x = _as_feature_matrix(context, "context")
    device, dtype = context.device, context.dtype
    left = bridge["linear_left"].to(device=device, dtype=dtype)
    right = bridge["linear_right"].to(device=device, dtype=dtype)
    correction = left @ (right.T @ context)

    keys = bridge.get("memory_keys")
    values = bridge.get("memory_values")
    if keys is not None and values is not None and keys.numel() and values.numel():
        query = torch.nn.functional.normalize(x, dim=0, eps=1e-8)
        keys = keys.to(device=device, dtype=torch.float32)
        values = values.to(device=device, dtype=dtype)
        similarity = keys.T @ query
        top_k = min(int(bridge["memory_top_k"]), similarity.shape[0])
        similarities, indices = similarity.topk(top_k, dim=0)
        temperature = float(bridge["memory_temperature"])
        weights = torch.softmax(similarities / temperature, dim=0).to(dtype=dtype)
        neighbors = values[:, indices.reshape(-1)].reshape(values.shape[0], top_k, x.shape[1])
        correction = correction + (neighbors * weights.unsqueeze(0)).sum(dim=1)
    return correction


def apply_defect_bridge_sequence(hidden_states: torch.Tensor, bridge: dict) -> torch.Tensor:
    """Evaluate a bridge on Hugging Face-style ``[..., features]`` states."""
    if not torch.is_tensor(hidden_states) or hidden_states.ndim < 2:
        raise ValueError("hidden_states must have shape [..., features]")
    feature_count = int(bridge["linear_right"].shape[0])
    if hidden_states.shape[-1] != feature_count:
        raise ValueError(
            f"hidden feature width {hidden_states.shape[-1]} does not match bridge width {feature_count}"
        )
    original_shape = hidden_states.shape
    context = hidden_states.reshape(-1, feature_count).T.contiguous()
    correction = apply_defect_bridge(context, bridge)
    return correction.T.reshape(original_shape)


def bridge_tensor_bytes(bridge: dict) -> int:
    """Count exact tensor payload bytes in a serialized bridge dictionary."""
    return sum(
        value.numel() * value.element_size()
        for value in bridge.values()
        if torch.is_tensor(value)
    )
