"""Propagated eigenspace error shields for initial NEC artifact construction.

PEES adds a small linear sidecar to selected NEC projections.  The sidecar is
fit from the dense-versus-compressed operator defect on the actual layer input
trajectories, then truncated in a ridge-whitened activation eigenspace.  It is
constructed before the artifact is serialized and never updates a saved
checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nox.block_compiler import RMSNorm, scaled_dot_product_attention_2d


DEFAULT_PROJECTIONS = ("v_proj", "o_proj", "gate_proj", "down_proj")


@torch.no_grad()
def layer_projection_trajectories(
    dense_weights: dict,
    compressed_weights: dict,
    dense_input: torch.Tensor,
    compressed_input: torch.Tensor,
    *,
    seq_len: int,
    n_heads: int,
    n_kv_heads: int,
    rms_norm_eps: float,
    max_tokens: int,
    projections=DEFAULT_PROJECTIONS,
):
    """Return matched dense/compressed projection contexts from sampled sequences.

    Sampling whole sequences preserves the causal attention context.  Sequence
    IDs are spread deterministically over the calibration stream so the sidecar
    uses training calibration data only and remains reproducible.
    """
    if dense_input.ndim != 2 or compressed_input.shape != dense_input.shape:
        raise ValueError("dense and compressed inputs must have matching [features, tokens] shapes")
    if seq_len <= 0 or dense_input.shape[1] % seq_len:
        raise ValueError("input token count must be divisible by seq_len")
    if max_tokens < seq_len:
        raise ValueError("max_tokens must include at least one full sequence")
    unknown = set(projections) - set(DEFAULT_PROJECTIONS)
    if unknown:
        raise ValueError(f"unsupported PEES projections: {sorted(unknown)}")

    total_sequences = dense_input.shape[1] // seq_len
    sequence_count = min(total_sequences, max(1, max_tokens // seq_len))
    sequence_ids = torch.linspace(
        0, total_sequences - 1, steps=sequence_count, device=dense_input.device,
    ).round().long().unique(sorted=True)
    token_ids = (
        sequence_ids[:, None] * seq_len
        + torch.arange(seq_len, device=dense_input.device)[None, :]
    ).reshape(-1)
    x_dense = dense_input.index_select(1, token_ids).contiguous()
    x_compressed = compressed_input.index_select(1, token_ids).contiguous()

    def contexts(weights, x):
        dim = x.shape[0]
        norm1 = RMSNorm(
            dim, eps=rms_norm_eps, weight=weights["input_layernorm"],
        )(x)
        q = weights["q_proj"] @ norm1
        k = weights["k_proj"] @ norm1
        v = weights["v_proj"] @ norm1
        attn = scaled_dot_product_attention_2d(
            q, k, v, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len,
        )
        residual = x + weights["o_proj"] @ attn
        norm2 = RMSNorm(
            dim, eps=rms_norm_eps, weight=weights["post_attention_layernorm"],
        )(residual)
        gate = weights["gate_proj"] @ norm2
        up = weights["up_proj"] @ norm2
        swiglu = F.silu(gate) * up
        return {
            "v_proj": norm1,
            "o_proj": attn,
            "gate_proj": norm2,
            "down_proj": swiglu,
        }

    dense_contexts = contexts(dense_weights, x_dense)
    compressed_contexts = contexts(compressed_weights, x_compressed)
    return {
        name: (dense_contexts[name], compressed_contexts[name])
        for name in projections
    }


def _randomized_svd(matrix: torch.Tensor, rank: int, *, oversample: int = 32, power_iterations: int = 1):
    """Compute a deterministic randomized truncated SVD on the matrix device."""
    rows, columns = matrix.shape
    keep = min(int(rank), rows, columns)
    if keep <= 0:
        raise ValueError("rank must be positive")
    sketch_rank = min(min(rows, columns), keep + int(oversample))
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(1729)
    omega = torch.randn(columns, sketch_rank, device=matrix.device, dtype=matrix.dtype, generator=generator)
    basis, _ = torch.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(max(0, int(power_iterations))):
        basis, _ = torch.linalg.qr(matrix @ (matrix.T @ basis), mode="reduced")
    small = basis.T @ matrix
    u_small, singular, vh = torch.linalg.svd(small, full_matrices=False)
    left = basis @ u_small[:, :keep]
    return left, singular[:keep], vh[:keep]


@torch.no_grad()
def fit_propagated_eigenspace_shield(
    dense_context: torch.Tensor,
    compressed_context: torch.Tensor,
    dense_weight: torch.Tensor,
    compressed_weight: torch.Tensor,
    *,
    rank: int,
    ridge_ratio: float = 0.01,
    storage_dtype: torch.dtype = torch.float16,
):
    """Fit a rank-r ridge-whitened low-rank correction to the propagated defect.

    With ``X`` the compressed context and ``R = W X_dense - W_hat X``, let
    ``X = U S V.T``.  Ridge regression gives the unconstrained correction
    ``C0 = R V S (S^2 + lambda I)^-1 U.T``.  PEES truncates the weighted
    operator ``R V S (S^2 + lambda I)^-1/2`` and maps its factors back through
    ``(S^2 + lambda I)^-1/2 U.T``.  The sidecar is therefore allocated to
    high-energy input eigendirections while its target includes both local
    approximation error and the incoming compressed-trajectory defect.

    The token Gram matrix is eigendecomposed instead of materializing an
    input-width covariance, which also supports the 11,008-wide SwiGLU context.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    if ridge_ratio < 0:
        raise ValueError("ridge_ratio must be nonnegative")
    if dense_context.ndim != 2 or compressed_context.ndim != 2:
        raise ValueError("contexts must be [features, samples] matrices")
    if dense_context.shape[1] != compressed_context.shape[1]:
        raise ValueError("dense and compressed contexts must share token columns")
    if dense_weight.ndim != 2 or compressed_weight.shape != dense_weight.shape:
        raise ValueError("dense and compressed weights must have matching matrix shapes")
    if dense_context.shape[0] != dense_weight.shape[1] or compressed_context.shape[0] != dense_weight.shape[1]:
        raise ValueError("context width does not match projection input width")
    if dense_context.shape[1] == 0:
        raise ValueError("at least one calibration token is required")

    device = compressed_context.device
    x_dense = dense_context.to(device=device, dtype=torch.float32)
    x = compressed_context.to(device=device, dtype=torch.float32)
    w = dense_weight.to(device=device, dtype=torch.float32)
    w_hat = compressed_weight.to(device=device, dtype=torch.float32)
    defect = w @ x_dense - w_hat @ x

    # X.T X is only N-by-N; N is the PEES calibration token cap, not the
    # projection width.  This avoids a large 11,008-by-11,008 Down covariance.
    token_gram = x.T @ x
    eigenvalues, right_singular = torch.linalg.eigh(token_gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    right_singular = right_singular[:, order]
    largest = eigenvalues[0].clamp_min(torch.finfo(eigenvalues.dtype).eps)
    active = eigenvalues > largest * 1e-8
    eigenvalues = eigenvalues[active]
    right_singular = right_singular[:, active]
    if eigenvalues.numel() == 0:
        raise RuntimeError("compressed context has zero numerical rank")

    singular = eigenvalues.sqrt()
    ridge = float(ridge_ratio) * (torch.trace(token_gram) / token_gram.shape[0]).clamp_min(1e-12)
    defect_modes = defect @ right_singular
    weighted_defect_modes = defect_modes * (
        singular / torch.sqrt(singular.square() + ridge)
    ).unsqueeze(0)
    effective_rank = min(int(rank), weighted_defect_modes.shape[0], weighted_defect_modes.shape[1])
    left, correction_singular, correction_vh = _randomized_svd(
        weighted_defect_modes, effective_rank,
    )

    # X V / S is the left singular basis in input space.  The ridge-whitened
    # right factor stays finite even for low-energy directions.
    input_basis = (x @ right_singular) / singular.unsqueeze(0)
    right = (
        correction_vh / torch.sqrt(singular.square() + ridge).unsqueeze(0)
    ) @ input_basis.T
    left = left * correction_singular.unsqueeze(0)

    # Match the actual FP16 artifact before reporting the calibration metrics.
    left_saved = left.to(dtype=storage_dtype)
    right_saved = right.to(dtype=storage_dtype)
    predicted = left_saved.float() @ (right_saved.float() @ x)
    defect_norm = torch.linalg.vector_norm(defect).clamp_min(1e-12)
    residual_norm = torch.linalg.vector_norm(defect - predicted)
    captured = (defect_modes.square().sum() / defect.square().sum().clamp_min(1e-12)).clamp(0, 1)
    metrics = {
        "objective": "propagated_defect_ridge_whitened_eigenspace",
        "requested_rank": int(rank),
        "effective_rank": int(effective_rank),
        "sample_tokens": int(x.shape[1]),
        "context_width": int(x.shape[0]),
        "ridge_ratio": float(ridge_ratio),
        "ridge": float(ridge),
        "calibration_relative_error_before": float(defect_norm / torch.linalg.vector_norm(w @ x_dense).clamp_min(1e-12)),
        "calibration_relative_error_after": float(residual_norm / torch.linalg.vector_norm(w @ x_dense).clamp_min(1e-12)),
        "defect_relative_residual_after": float(residual_norm / defect_norm),
        "attainable_defect_energy_fraction": float(captured),
        "correction_to_defect_norm_ratio": float(torch.linalg.vector_norm(predicted) / defect_norm),
        "serialized_tensor_bytes": int(left_saved.numel() * left_saved.element_size() + right_saved.numel() * right_saved.element_size()),
    }
    return left_saved.cpu(), right_saved.cpu(), metrics


def append_shield_to_projection_artifact(artifact: dict, left: torch.Tensor, right: torch.Tensor) -> int:
    """Add PEES factors to the existing bridge sum without changing MPO cores."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("PEES factors must have shapes [d_out, rank] and [rank, d_in]")
    if tuple(artifact.get("orig_shape", ())) != (left.shape[0], right.shape[1]):
        raise ValueError("PEES factor shapes do not match the saved projection")
    old_left, old_right = artifact.get("A_bridge"), artifact.get("B_bridge")
    if old_left is None and old_right is None:
        artifact["A_bridge"], artifact["B_bridge"] = left, right
    elif old_left is not None and old_right is not None:
        artifact["A_bridge"] = torch.cat((old_left, left), dim=1)
        artifact["B_bridge"] = torch.cat((old_right, right), dim=0)
    else:
        raise ValueError("existing NEC bridge must contain both factors or neither")
    return int(left.numel() * left.element_size() + right.numel() * right.element_size())
