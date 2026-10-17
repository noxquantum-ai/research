"""Joint shared-neuron tensor-network representation for a SwiGLU MLP.

This opt-in diagnostic representation couples Gate, Up, and Down along their
shared intermediate-neuron axis.  It is deliberately separate from the
per-projection NEC default until a held-out function-level experiment supports
it.

For a calibration matrix X, form the neuron-state matrix

    S = [G X, U X, SiLU(G X) * (U X)] .

The leading left singular vectors Z_r are the optimal rank-r common orthogonal
projector for the sum of squared calibration activation residuals across these
three states.  The three leaves then share that bond:

    G ~= Z_r G_r,  U ~= Z_r U_r,  D ~= D_r Z_r^T.

This is a three-arm Tucker/tree tensor network on the SwiGLU neuron index.  It
uses m*r + 3*d*r floating coefficients, has no dense residual, and applies the
whole nonlinearity with factorized operators.  The SVD objective is only a
construction surrogate; held-out complete-MLP error remains the decision
metric.
"""

from __future__ import annotations

import io
import itertools

import torch
import torch.nn.functional as F


CODEC_TYPE = "nec_joint_swiglu_shared_neuron_ttn_v1"
ROLE_ORDER = ("gate_preactivation", "up_preactivation", "swiglu_product")


def joint_neuron_coefficient_count(d_model: int, intermediate_size: int, rank: int) -> int:
    """Number of stored floating coefficients in the shared three-arm TTN."""
    d_model = int(d_model)
    intermediate_size = int(intermediate_size)
    rank = int(rank)
    if min(d_model, intermediate_size, rank) <= 0:
        raise ValueError("dimensions and rank must be positive")
    return rank * (intermediate_size + 3 * d_model)


def joint_neuron_rank_for_budget(
    coefficient_budget: int,
    d_model: int,
    intermediate_size: int,
    *,
    calibration_tokens: int | None = None,
) -> int:
    """Largest nonzero shared rank fitting a floating-coefficient budget."""
    coefficient_budget = int(coefficient_budget)
    d_model = int(d_model)
    intermediate_size = int(intermediate_size)
    if coefficient_budget <= 0 or min(d_model, intermediate_size) <= 0:
        raise ValueError("budget and dimensions must be positive")
    max_rank = min(intermediate_size, 3 * d_model)
    if calibration_tokens is not None:
        if int(calibration_tokens) <= 0:
            raise ValueError("calibration_tokens must be positive")
        max_rank = min(max_rank, 3 * int(calibration_tokens))
    rank = min(
        max_rank,
        coefficient_budget // (intermediate_size + 3 * d_model),
    )
    if rank < 1:
        raise ValueError("coefficient budget is too small for one shared neuron rank")
    return rank


def _left_singular_vectors(
    matrix: torch.Tensor,
    rank: int,
    *,
    oversample: int = 16,
    power_iterations: int = 2,
    seed: int = 0,
) -> torch.Tensor:
    """Deterministic truncated left SVD, randomized for large activation stacks."""
    if matrix.ndim != 2:
        raise ValueError("activation state matrix must be rank two")
    source = matrix.detach().to(dtype=torch.float32)
    if not torch.isfinite(source).all():
        raise ValueError("activation state matrix contains non-finite values")
    rows, columns = source.shape
    rank = min(int(rank), rows, columns)
    if rank <= 0:
        raise ValueError("rank must be positive and no larger than the state matrix")
    if source.numel() <= 2_000_000 or rank == min(rows, columns):
        left, _, _ = torch.linalg.svd(source, full_matrices=False)
        return left[:, :rank].contiguous()

    sketch_rank = min(rank + max(0, int(oversample)), rows, columns)
    generator = torch.Generator(device=source.device)
    generator.manual_seed(int(seed))
    omega = torch.randn(
        columns, sketch_rank, device=source.device, dtype=source.dtype,
        generator=generator,
    )
    basis = torch.linalg.qr(source @ omega, mode="reduced").Q
    for _ in range(max(0, int(power_iterations))):
        basis = torch.linalg.qr(source @ (source.T @ basis), mode="reduced").Q
    small = basis.T @ source
    left_small, _, _ = torch.linalg.svd(small, full_matrices=False)
    return (basis @ left_small[:, :rank]).contiguous()


@torch.no_grad()
def build_joint_swiglu_artifact(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    calibration_input: torch.Tensor,
    *,
    rank: int | None = None,
    coefficient_budget: int | None = None,
    storage_dtype: torch.dtype | None = None,
    oversample: int = 16,
    power_iterations: int = 2,
    seed: int = 0,
) -> dict:
    """Construct a fixed-budget shared-neuron TTN from dense MLP weights.

    Matrix shapes follow Llama's linear convention: Gate/Up are [m, d], Down
    is [d, m], and calibration_input is [d, n].  The basis uses calibration
    activations only; no validation data or optimization loop is accepted.
    """
    if gate_weight.ndim != 2 or up_weight.ndim != 2 or down_weight.ndim != 2:
        raise ValueError("Gate, Up, and Down weights must be matrices")
    if gate_weight.shape != up_weight.shape:
        raise ValueError("Gate and Up weights must have identical shapes")
    intermediate_size, d_model = gate_weight.shape
    if down_weight.shape != (d_model, intermediate_size):
        raise ValueError("Down weight must have shape [d_model, intermediate_size]")
    if calibration_input.ndim != 2 or calibration_input.shape[0] != d_model:
        raise ValueError("calibration_input must have shape [d_model, n_tokens]")
    if calibration_input.shape[1] <= 0:
        raise ValueError("calibration_input must contain at least one token")
    if (rank is None) == (coefficient_budget is None):
        raise ValueError("provide exactly one of rank or coefficient_budget")
    if rank is None:
        rank = joint_neuron_rank_for_budget(
            coefficient_budget, d_model, intermediate_size,
            calibration_tokens=calibration_input.shape[1],
        )
    rank = int(rank)
    max_rank = min(intermediate_size, 3 * d_model, 3 * calibration_input.shape[1])
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}] for these dimensions")
    if coefficient_budget is not None:
        required = joint_neuron_coefficient_count(d_model, intermediate_size, rank)
        if required > int(coefficient_budget):
            raise ValueError("selected rank exceeds coefficient_budget")

    device = gate_weight.device
    tensors = (up_weight, down_weight, calibration_input)
    if any(value.device != device for value in tensors):
        raise ValueError("all weights and calibration_input must share a device")
    gate = gate_weight.detach().float()
    up = up_weight.detach().float()
    down = down_weight.detach().float()
    inputs = calibration_input.detach().float()
    if not all(torch.isfinite(value).all() for value in (gate, up, down, inputs)):
        raise ValueError("weights and calibration_input must be finite")

    gate_states = gate @ inputs
    up_states = up @ inputs
    product_states = F.silu(gate_states) * up_states
    joint_states = torch.cat((gate_states, up_states, product_states), dim=1)
    basis = _left_singular_vectors(
        joint_states, rank, oversample=oversample,
        power_iterations=power_iterations, seed=seed,
    )

    target_dtype = storage_dtype or gate_weight.dtype
    artifact = {
        "codec_type": CODEC_TYPE,
        "role_order": ROLE_ORDER,
        "orig_shapes": {
            "gate": tuple(gate_weight.shape),
            "up": tuple(up_weight.shape),
            "down": tuple(down_weight.shape),
        },
        "rank": rank,
        "neuron_basis": basis.to(dtype=target_dtype),
        "gate_factor": (basis.T @ gate).to(dtype=target_dtype),
        "up_factor": (basis.T @ up).to(dtype=target_dtype),
        "down_factor": (down @ basis).to(dtype=target_dtype),
        "coefficient_count": joint_neuron_coefficient_count(
            d_model, intermediate_size, rank,
        ),
        "construction_objective": "joint_calibration_activation_subspace",
        "construction_uses_validation": False,
        "post_compression_optimization": False,
        "gradient_healing": False,
    }
    actual_count = sum(
        artifact[name].numel()
        for name in ("neuron_basis", "gate_factor", "up_factor", "down_factor")
    )
    if actual_count != artifact["coefficient_count"]:
        raise RuntimeError("internal shared-neuron coefficient accounting mismatch")
    return artifact


def _validate_artifact(artifact: dict) -> tuple[int, int, int]:
    if artifact.get("codec_type") != CODEC_TYPE:
        raise ValueError(f"unsupported shared-SwiGLU artifact: {artifact.get('codec_type')!r}")
    basis = artifact["neuron_basis"]
    gate = artifact["gate_factor"]
    up = artifact["up_factor"]
    down = artifact["down_factor"]
    if basis.ndim != 2 or gate.ndim != 2 or up.ndim != 2 or down.ndim != 2:
        raise ValueError("shared-SwiGLU factors must be rank-two tensors")
    intermediate_size, rank = basis.shape
    if gate.shape[0] != rank or up.shape[0] != rank or down.shape[1] != rank:
        raise ValueError("shared-SwiGLU factor ranks do not agree")
    d_model = gate.shape[1]
    if up.shape[1] != d_model or down.shape[0] != d_model:
        raise ValueError("shared-SwiGLU factor dimensions do not agree")
    if tuple(artifact.get("orig_shapes", {}).get("gate", ())) != (intermediate_size, d_model):
        raise ValueError("shared-SwiGLU original shapes do not match the stored factors")
    return d_model, intermediate_size, rank


def reconstruct_joint_swiglu_weights(artifact: dict, *, device="cpu", dtype=torch.float32):
    """Reconstruct Gate/Up/Down matrices for diagnostics and dense injection."""
    _validate_artifact(artifact)
    target = torch.device(device)
    basis = artifact["neuron_basis"].to(device=target, dtype=dtype)
    gate_factor = artifact["gate_factor"].to(device=target, dtype=dtype)
    up_factor = artifact["up_factor"].to(device=target, dtype=dtype)
    down_factor = artifact["down_factor"].to(device=target, dtype=dtype)
    return {
        "gate_proj": basis @ gate_factor,
        "up_proj": basis @ up_factor,
        "down_proj": down_factor @ basis.T,
    }


@torch.no_grad()
def apply_joint_swiglu(artifact: dict, inputs: torch.Tensor) -> torch.Tensor:
    """Apply the complete represented SwiGLU MLP without dense reconstruction."""
    d_model, intermediate_size, _ = _validate_artifact(artifact)
    if inputs.ndim != 2 or inputs.shape[0] != d_model:
        raise ValueError(f"inputs must have shape [{d_model}, n_tokens]")
    target_dtype = inputs.dtype
    device = inputs.device
    x = inputs
    basis = artifact["neuron_basis"].to(device=device, dtype=target_dtype)
    gate_factor = artifact["gate_factor"].to(device=device, dtype=target_dtype)
    up_factor = artifact["up_factor"].to(device=device, dtype=target_dtype)
    down_factor = artifact["down_factor"].to(device=device, dtype=target_dtype)
    if basis.shape[0] != intermediate_size:
        raise ValueError("shared neuron dimension does not match input factors")
    gate = basis @ (gate_factor @ x)
    up = basis @ (up_factor @ x)
    product = F.silu(gate) * up
    return down_factor @ (basis.T @ product)


def shared_swiglu_payload_bytes(artifact: dict) -> int:
    _validate_artifact(artifact)
    return int(sum(
        artifact[name].numel() * artifact[name].element_size()
        for name in ("neuron_basis", "gate_factor", "up_factor", "down_factor")
    ))


def shared_swiglu_serialized_bytes(artifact: dict) -> int:
    stream = io.BytesIO()
    torch.save(artifact, stream)
    return stream.tell()


def _relative_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref = reference.float()
    cand = candidate.float()
    error = cand - ref
    ref_norm = torch.linalg.vector_norm(ref).clamp_min(1e-30)
    error_norm = torch.linalg.vector_norm(error)
    return {
        "relative_error": float(error_norm / ref_norm),
        "reference_norm": float(ref_norm),
        "error_norm": float(error_norm),
    }


def _term_interference(left: torch.Tensor, right: torch.Tensor) -> dict:
    left = left.float()
    right = right.float()
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    dot = torch.sum(left * right)
    energy_sum = left_norm.square() + right_norm.square()
    return {
        "cosine": float(dot / (left_norm * right_norm).clamp_min(1e-30)),
        "cross_energy_fraction": float(2.0 * dot / energy_sum.clamp_min(1e-30)),
    }


@torch.no_grad()
def mlp_error_decomposition(
    dense_gate: torch.Tensor,
    dense_up: torch.Tensor,
    dense_down: torch.Tensor,
    compressed_gate: torch.Tensor,
    compressed_up: torch.Tensor,
    compressed_down: torch.Tensor,
    inputs: torch.Tensor,
) -> dict:
    """Measure the exact Gate/Up product and Down error decomposition.

    The principal identity is

        D_hat h_hat - D h = D (h_hat - h) + (D_hat - D) h_hat.

    The report also exposes the nonlinear Gate-only, Up-only, and bilinear
    interaction terms.  All eight dense/compressed projection combinations
    are evaluated on the same fixed inputs.
    """
    if any(value.ndim != 2 for value in (
        dense_gate, dense_up, dense_down, compressed_gate, compressed_up,
        compressed_down, inputs,
    )):
        raise ValueError("weights and inputs must all be matrices")
    if dense_gate.shape != dense_up.shape or dense_gate.shape != compressed_gate.shape:
        raise ValueError("dense/compressed Gate and Up shapes must match")
    intermediate_size, d_model = dense_gate.shape
    if dense_down.shape != (d_model, intermediate_size):
        raise ValueError("Down weight has incompatible shape")
    if compressed_down.shape != dense_down.shape:
        raise ValueError("compressed Down weight has incompatible shape")
    if inputs.shape[0] != d_model:
        raise ValueError("input width does not match Gate/Up")
    device = inputs.device
    if any(value.device != device for value in (
        dense_gate, dense_up, dense_down, compressed_gate, compressed_up,
        compressed_down,
    )):
        raise ValueError("all weights and inputs must share a device")

    dense_g = dense_gate.float() @ inputs.float()
    dense_u = dense_up.float() @ inputs.float()
    compressed_g = compressed_gate.float() @ inputs.float()
    compressed_u = compressed_up.float() @ inputs.float()
    dense_activation = F.silu(dense_g)
    compressed_activation = F.silu(compressed_g)
    dense_product = dense_activation * dense_u
    compressed_product = compressed_activation * compressed_u
    dense_output = dense_down.float() @ dense_product

    gate_silu_delta = compressed_activation - dense_activation
    up_delta = compressed_u - dense_u
    gate_hidden_delta = gate_silu_delta * dense_u
    up_hidden_delta = dense_activation * up_delta
    cross_hidden_delta = gate_silu_delta * up_delta
    gate_up_error_terms = {
        "gate_only": dense_down.float() @ gate_hidden_delta,
        "up_only": dense_down.float() @ up_hidden_delta,
        "gate_up_interaction": dense_down.float() @ cross_hidden_delta,
    }
    gate_up_sum = sum(gate_up_error_terms.values())
    joint_product_delta = dense_down.float() @ (compressed_product - dense_product)
    product_term_identity = torch.linalg.vector_norm(
        gate_up_sum - joint_product_delta
    ) / torch.linalg.vector_norm(joint_product_delta).clamp_min(1e-30)
    gate_up_names = tuple(gate_up_error_terms)
    pairwise_interference = {}
    for left, right in itertools.combinations(gate_up_names, 2):
        pairwise_interference[f"{left}__{right}"] = _term_interference(
            gate_up_error_terms[left], gate_up_error_terms[right],
        )

    combinations = {}
    for gate_source, up_source, down_source in itertools.product(
        ("dense", "compressed"), repeat=3,
    ):
        selected_gate = dense_g if gate_source == "dense" else compressed_g
        selected_up = dense_u if up_source == "dense" else compressed_u
        selected_down = dense_down if down_source == "dense" else compressed_down
        selected_product = F.silu(selected_gate) * selected_up
        selected_output = selected_down.float() @ selected_product
        product_error = selected_product - dense_product
        product_output_error = dense_down.float() @ product_error
        down_output_error = (
            selected_down.float() - dense_down.float()
        ) @ selected_product
        summed_error = selected_output - dense_output
        reconstruction_error = summed_error - product_output_error - down_output_error
        decomposition_scale = (
            torch.linalg.vector_norm(product_output_error)
            + torch.linalg.vector_norm(down_output_error)
        ).clamp_min(torch.linalg.vector_norm(dense_output) * 1e-7 + 1e-30)
        combinations[f"gate_{gate_source}__up_{up_source}__down_{down_source}"] = {
            "gate_source": gate_source,
            "up_source": up_source,
            "down_source": down_source,
            "gate_preactivation_relative_error": _relative_metrics(
                dense_g, selected_gate,
            )["relative_error"],
            "up_preactivation_relative_error": _relative_metrics(
                dense_u, selected_up,
            )["relative_error"],
            "swiglu_product_relative_error": _relative_metrics(
                dense_product, selected_product,
            )["relative_error"],
            "mlp_output_relative_error": _relative_metrics(
                dense_output, selected_output,
            )["relative_error"],
            "product_error_after_dense_down_relative": float(
                torch.linalg.vector_norm(product_output_error)
                / torch.linalg.vector_norm(dense_output).clamp_min(1e-30)
            ),
            "down_error_on_selected_product_relative": float(
                torch.linalg.vector_norm(down_output_error)
                / torch.linalg.vector_norm(dense_output).clamp_min(1e-30)
            ),
            "product_down_error_cosine": float(
                F.cosine_similarity(
                    product_output_error.reshape(1, -1),
                    down_output_error.reshape(1, -1),
                    dim=1, eps=1e-30,
                )[0]
            ),
            "product_down_cross_energy_fraction": _term_interference(
                product_output_error, down_output_error,
            )["cross_energy_fraction"],
            "decomposition_identity_relative_residual": float(
                torch.linalg.vector_norm(reconstruction_error)
                / decomposition_scale
            ),
        }

    return {
        "decomposition": "D_hat*h_hat - D*h = D*(h_hat-h) + (D_hat-D)*h_hat",
        "dense_swiglu_product_norm": float(torch.linalg.vector_norm(dense_product)),
        "compressed_swiglu_product_relative_error": _relative_metrics(
            dense_product, compressed_product,
        )["relative_error"],
        "compressed_product_through_dense_down_relative_error": _relative_metrics(
            dense_output, dense_down.float() @ compressed_product,
        )["relative_error"],
        "gate_up_product_terms": {
            name: {
                "relative_to_dense_mlp_output": float(
                    torch.linalg.vector_norm(value)
                    / torch.linalg.vector_norm(dense_output).clamp_min(1e-30)
                ),
                "norm": float(torch.linalg.vector_norm(value)),
            }
            for name, value in gate_up_error_terms.items()
        },
        "gate_up_pairwise_interference": pairwise_interference,
        "gate_up_product_identity_relative_residual": float(product_term_identity),
        "combinations": combinations,
    }
