"""CPU-friendly diagnostics for the shared-neuron SwiGLU rank bottleneck.

These routines do not construct or optimize a compression representation. They
compute (1) the Eckart--Young output-matrix rank floor and (2) controlled
Gate/Up with alternate Down-operator crossovers for an already constructed
candidate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F


def relative_rank_tail_errors(
    singular_values: torch.Tensor,
    ranks: Iterable[int],
) -> dict[int, float]:
    """Return optimal relative Frobenius error for each requested matrix rank.

    The input may be the singular values of any matrix, sorted or unsorted.
    Values are sorted internally, accumulated in float64, and rank zero is
    supported. This is the Eckart--Young lower bound for *any* rank-r matrix
    approximation of the sampled output matrix; it is not a bound for all
    tensor-network representations.
    """
    values = torch.as_tensor(singular_values).detach().to(device="cpu", dtype=torch.float64)
    if values.ndim != 1:
        raise ValueError("singular_values must be one-dimensional")
    if values.numel() == 0:
        raise ValueError("singular_values must be non-empty")
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError("singular_values must be finite and nonnegative")
    values = values.sort(descending=True).values
    energy = values.square()
    total = energy.sum()
    if total <= 0:
        raise ValueError("singular_values must contain positive energy")
    retained = energy.cumsum(0)
    output: dict[int, float] = {}
    for raw_rank in ranks:
        rank = int(raw_rank)
        if rank < 0:
            raise ValueError("ranks must be nonnegative")
        clipped = min(rank, values.numel())
        tail = total if clipped == 0 else total - retained[clipped - 1]
        output[rank] = math.sqrt(max(0.0, float(tail / total)))
    return output


@torch.no_grad()
def output_rank_curve(
    dense_output: torch.Tensor,
    ranks: Iterable[int],
    *,
    device: str | torch.device = "cpu",
) -> dict:
    """Compute singular values and the Eckart--Young tail curve.

    A float64 SVD is used deliberately: this diagnostic is a lower-bound
    measurement, where numerical stability is more important than speed. The
    SVD can run on CUDA; values are copied to CPU once for tail accounting.
    """
    if dense_output.ndim != 2:
        raise ValueError("dense_output must have shape [output_dim, n_samples]")
    matrix = dense_output.detach().to(device=torch.device(device), dtype=torch.float64)
    if not torch.isfinite(matrix).all():
        raise ValueError("dense_output contains non-finite values")
    singular_values = torch.linalg.svdvals(matrix).to(device="cpu")
    requested = [int(rank) for rank in ranks]
    errors = relative_rank_tail_errors(singular_values, requested)
    numerical_rank = int(torch.count_nonzero(singular_values > (
        singular_values.max() * max(matrix.shape) * torch.finfo(torch.float64).eps
    )))
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "numerical_rank": numerical_rank,
        "frobenius_norm": float(torch.linalg.vector_norm(matrix).cpu()),
        "singular_values": [float(value) for value in singular_values],
        "relative_tail_error_by_rank": {str(rank): error for rank, error in errors.items()},
    }


def swiglu_output(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate D[SiLU(GX) ⊙ (UX)] for matrices stored as [out, in]."""
    if any(value.ndim != 2 for value in (gate_weight, up_weight, down_weight, inputs)):
        raise ValueError("weights and inputs must be rank-two tensors")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("Gate and Up weights must have identical shapes")
    intermediate, d_model = gate_weight.shape
    if down_weight.shape != (d_model, intermediate) or inputs.shape[0] != d_model:
        raise ValueError("incompatible Down or input dimensions")
    gate = gate_weight @ inputs
    up = up_weight @ inputs
    return down_weight @ (F.silu(gate) * up)


def _relative_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(estimate.float() - reference.float()) / denominator)


def _interference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left = left.float()
    right = right.float()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    denominator = left_norm.square() + right_norm.square()
    return {
        "cosine": float(torch.sum(left * right) / (left_norm * right_norm).clamp_min(1e-30)),
        "cross_energy_fraction": float(2.0 * torch.sum(left * right) / denominator.clamp_min(1e-30)),
    }


@torch.no_grad()
def evaluate_down_crossovers(
    *,
    dense_gate: torch.Tensor,
    dense_up: torch.Tensor,
    dense_down: torch.Tensor,
    candidate_gate: torch.Tensor,
    candidate_up: torch.Tensor,
    candidate_down: torch.Tensor,
    saved_nec_gate: torch.Tensor,
    saved_nec_up: torch.Tensor,
    saved_nec_down: torch.Tensor,
    inputs: torch.Tensor,
) -> dict:
    """Evaluate baseline and candidate G/U with dense, saved, and candidate D.

    The dense-Down branch is an oracle diagnostic only. All three candidate
    branches use the same candidate Gate/Up product, so their error
    differences isolate the contribution of the Down representation.
    """
    target = swiglu_output(dense_gate, dense_up, dense_down, inputs)
    dense_product = F.silu(dense_gate @ inputs) * (dense_up @ inputs)
    candidate_product = F.silu(candidate_gate @ inputs) * (candidate_up @ inputs)
    saved_product = F.silu(saved_nec_gate @ inputs) * (saved_nec_up @ inputs)

    rows = {}
    for name, down in (
        ("dense_down_oracle", dense_down),
        ("saved_nec_down", saved_nec_down),
        ("candidate_down", candidate_down),
    ):
        estimate = down @ candidate_product
        product_term = dense_down @ (candidate_product - dense_product)
        down_term = (down - dense_down) @ candidate_product
        identity_residual = estimate - target - product_term - down_term
        rows[name] = {
            "mlp_output_relative_error": _relative_error(target, estimate),
            "product_relative_error": _relative_error(dense_product, candidate_product),
            "product_error_after_dense_down_relative": _relative_error(target, dense_down @ candidate_product),
            "product_term_relative_to_dense_output": _relative_error(target, product_term + target),
            "down_term_relative_to_dense_output": float(
                torch.linalg.vector_norm(down_term.float())
                / torch.linalg.vector_norm(target.float()).clamp_min(1e-30)
            ),
            "product_down_interference": _interference(product_term, down_term),
            "decomposition_identity_relative_residual": float(
                torch.linalg.vector_norm(identity_residual.float())
                / torch.linalg.vector_norm(target.float()).clamp_min(1e-30)
            ),
        }

    saved_estimate = saved_nec_down @ saved_product
    rows["saved_nec_baseline"] = {
        "mlp_output_relative_error": _relative_error(target, saved_estimate),
        "product_relative_error": _relative_error(dense_product, saved_product),
    }
    return rows
