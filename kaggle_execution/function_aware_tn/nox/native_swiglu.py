"""Native-coordinate structural compression for a complete SwiGLU module.

The coordinatewise SwiGLU nonlinearity is preserved exactly for a selected
subset of original intermediate neurons:

    y_hat(x) = B_S [silu(G_S x) * (U_S x)].

Unlike a shared low-rank neuron projector, this representation never mixes the
neuron axis across the nonlinearity.  G_S and U_S are selected rows of the
dense operators.  B_S is initialized from the matching Down columns and can
be constructed once with a calibration-only, prior-anchored ridge solve.
There is no dense residual and inference applies the stored rectangular
operators directly.
"""

from __future__ import annotations

import io

import torch
import torch.nn.functional as F


CODEC_TYPE = "nec_native_swiglu_v1"


def native_swiglu_coefficient_count(
    d_model: int, retained_neurons: int,
) -> int:
    """Floating coefficients for Gate, Up, and Down rectangular operators."""
    d_model = int(d_model)
    retained_neurons = int(retained_neurons)
    if d_model <= 0 or retained_neurons <= 0:
        raise ValueError("dimensions must be positive")
    return 3 * d_model * retained_neurons


def retained_neurons_for_fraction(intermediate_size: int, fraction: float) -> int:
    intermediate_size = int(intermediate_size)
    fraction = float(fraction)
    if intermediate_size <= 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("intermediate_size must be positive and fraction in (0, 1]")
    return max(1, min(intermediate_size, int(round(intermediate_size * fraction))))


def _validate_weights(gate, up, down, inputs) -> tuple[int, int]:
    if any(value.ndim != 2 for value in (gate, up, down, inputs)):
        raise ValueError("weights and calibration inputs must be matrices")
    if gate.shape != up.shape:
        raise ValueError("Gate and Up weights must have identical shapes")
    intermediate_size, d_model = gate.shape
    if down.shape != (d_model, intermediate_size):
        raise ValueError("Down must have shape [d_model, intermediate_size]")
    if inputs.shape[0] != d_model or inputs.shape[1] <= 0:
        raise ValueError("inputs must have shape [d_model, positive_tokens]")
    return d_model, intermediate_size


@torch.no_grad()
def native_neuron_importance(gate, up, down, calibration_input) -> torch.Tensor:
    """Output-energy score for each original SwiGLU neuron.

    For h_j(x)=silu(g_j x)(u_j x), the squared norm of the individual output
    contribution d_j h_j is ||d_j||^2 E[h_j^2].  This preserves the actual
    nonlinear feature and its Down influence, rather than scoring three weight
    matrices independently.
    """
    _validate_weights(gate, up, down, calibration_input)
    gate_state = gate.float() @ calibration_input.float()
    up_state = up.float() @ calibration_input.float()
    feature_energy = (F.silu(gate_state) * up_state).square().mean(dim=1)
    down_energy = down.float().square().sum(dim=0)
    score = feature_energy * down_energy
    if not torch.isfinite(score).all():
        raise ValueError("non-finite native-neuron importance")
    return score


@torch.no_grad()
def build_native_swiglu_artifact(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    calibration_input: torch.Tensor,
    *,
    retained_neurons: int | None = None,
    retain_fraction: float | None = None,
    storage_dtype: torch.dtype = torch.float16,
) -> dict:
    """Select native neurons using calibration inputs and build the artifact."""
    d_model, intermediate_size = _validate_weights(
        gate_weight, up_weight, down_weight, calibration_input,
    )
    if (retained_neurons is None) == (retain_fraction is None):
        raise ValueError("provide exactly one of retained_neurons or retain_fraction")
    if retained_neurons is None:
        retained_neurons = retained_neurons_for_fraction(
            intermediate_size, float(retain_fraction),
        )
    retained_neurons = int(retained_neurons)
    if not 1 <= retained_neurons <= intermediate_size:
        raise ValueError("retained_neurons is outside the valid range")

    score = native_neuron_importance(
        gate_weight, up_weight, down_weight, calibration_input,
    )
    # Sorting selected indices restores native order and makes serialization
    # deterministic without changing which neurons are retained.
    indices = torch.topk(score, retained_neurons, sorted=False).indices.sort().values
    artifact = {
        "codec_type": CODEC_TYPE,
        "orig_shapes": {
            "gate": tuple(gate_weight.shape),
            "up": tuple(up_weight.shape),
            "down": tuple(down_weight.shape),
        },
        "retained_neurons": retained_neurons,
        "retained_indices": indices.to(device="cpu", dtype=torch.int32),
        "gate_rows": gate_weight.index_select(0, indices).to(
            device="cpu", dtype=storage_dtype,
        ),
        "up_rows": up_weight.index_select(0, indices).to(
            device="cpu", dtype=storage_dtype,
        ),
        "down_columns": down_weight.index_select(1, indices).to(
            device="cpu", dtype=storage_dtype,
        ),
        "coefficient_count": native_swiglu_coefficient_count(
            d_model, retained_neurons,
        ),
        "construction_objective": "native_neuron_output_contribution",
        "construction_uses_validation": False,
        "post_compression_optimization": False,
        "uses_dense_residual": False,
    }
    _validate_artifact(artifact)
    return artifact


def _validate_artifact(artifact: dict) -> tuple[int, int, int]:
    if artifact.get("codec_type") != CODEC_TYPE:
        raise ValueError(f"unsupported native SwiGLU artifact: {artifact.get('codec_type')!r}")
    gate = artifact["gate_rows"]
    up = artifact["up_rows"]
    down = artifact["down_columns"]
    indices = artifact["retained_indices"]
    if gate.ndim != 2 or up.shape != gate.shape or down.ndim != 2:
        raise ValueError("invalid native SwiGLU factor shapes")
    retained, d_model = gate.shape
    if down.shape != (d_model, retained) or indices.shape != (retained,):
        raise ValueError("native SwiGLU factors disagree")
    intermediate_size = int(artifact["orig_shapes"]["gate"][0])
    if tuple(artifact["orig_shapes"]["gate"]) != (intermediate_size, d_model):
        raise ValueError("invalid original Gate shape")
    if tuple(artifact["orig_shapes"]["up"]) != (intermediate_size, d_model):
        raise ValueError("invalid original Up shape")
    if tuple(artifact["orig_shapes"]["down"]) != (d_model, intermediate_size):
        raise ValueError("invalid original Down shape")
    if retained != int(artifact["retained_neurons"]):
        raise ValueError("retained-neuron metadata mismatch")
    if int(artifact["coefficient_count"]) != native_swiglu_coefficient_count(d_model, retained):
        raise ValueError("coefficient accounting mismatch")
    return d_model, intermediate_size, retained


@torch.no_grad()
def native_swiglu_features(artifact: dict, inputs: torch.Tensor) -> torch.Tensor:
    d_model, _, _ = _validate_artifact(artifact)
    if inputs.ndim != 2 or inputs.shape[0] != d_model:
        raise ValueError(f"inputs must have shape [{d_model}, tokens]")
    dtype = inputs.dtype
    gate = artifact["gate_rows"].to(device=inputs.device, dtype=dtype)
    up = artifact["up_rows"].to(device=inputs.device, dtype=dtype)
    return F.silu(gate @ inputs) * (up @ inputs)


@torch.no_grad()
def apply_native_swiglu(artifact: dict, inputs: torch.Tensor) -> torch.Tensor:
    features = native_swiglu_features(artifact, inputs)
    down = artifact["down_columns"].to(device=inputs.device, dtype=inputs.dtype)
    return down @ features


def native_swiglu_payload_bytes(artifact: dict) -> int:
    _validate_artifact(artifact)
    return int(sum(
        artifact[name].numel() * artifact[name].element_size()
        for name in ("retained_indices", "gate_rows", "up_rows", "down_columns")
    ))


def native_swiglu_serialized_bytes(artifact: dict) -> int:
    _validate_artifact(artifact)
    stream = io.BytesIO()
    torch.save(artifact, stream)
    return int(stream.tell())


class NativeSwiGLUModule(torch.nn.Module):
    """Executable Hugging Face-compatible module backed by a native artifact."""

    def __init__(self, artifact: dict, *, device=None, dtype=None):
        super().__init__()
        _validate_artifact(artifact)
        gate = artifact["gate_rows"]
        target_dtype = dtype or gate.dtype
        self.register_buffer(
            "gate_rows", gate.to(device=device, dtype=target_dtype), persistent=True,
        )
        self.register_buffer(
            "up_rows", artifact["up_rows"].to(device=device, dtype=target_dtype),
            persistent=True,
        )
        self.register_buffer(
            "down_columns",
            artifact["down_columns"].to(device=device, dtype=target_dtype),
            persistent=True,
        )
        self.register_buffer(
            "retained_indices", artifact["retained_indices"].to(device=device),
            persistent=True,
        )
        self.intermediate_size = int(artifact["orig_shapes"]["gate"][0])
        self.retained_neurons = int(artifact["retained_neurons"])
        self.codec_type = CODEC_TYPE

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.linear(hidden_states, self.gate_rows)
        up = F.linear(hidden_states, self.up_rows)
        return F.linear(F.silu(gate) * up, self.down_columns)


def _sequence_split_indices(tokens: int, sequence_length: int, fit_fraction: float):
    if tokens % sequence_length:
        raise ValueError("tokens must be divisible by sequence_length")
    sequences = tokens // sequence_length
    fit_sequences = max(1, min(sequences - 1, int(round(sequences * fit_fraction))))
    boundary = fit_sequences * sequence_length
    return slice(0, boundary), slice(boundary, tokens)


@torch.no_grad()
def _anchored_validation_prediction(
    fit_features: torch.Tensor,
    fit_targets: torch.Tensor,
    validation_features: torch.Tensor,
    prior: torch.Tensor,
    ridge_ratio: float,
) -> torch.Tensor:
    """Evaluate an anchored ridge readout without materializing its full matrix."""
    h = fit_features.float()
    y = fit_targets.float()
    hv = validation_features.float()
    b0 = prior.float()
    residual = y - b0 @ h
    gram = h.T @ h
    scale = gram.diagonal().mean().clamp_min(1e-8)
    ridge = max(float(ridge_ratio), 0.0) * scale
    gram.diagonal().add_(ridge + scale * 1e-7)
    cross = h.T @ hv
    solved = torch.linalg.solve(gram, cross)
    return b0 @ hv + residual @ solved


@torch.no_grad()
def fit_anchored_down_readout(
    features: torch.Tensor,
    targets: torch.Tensor,
    prior: torch.Tensor,
    *,
    sequence_length: int = 128,
    ridge_ratios=(1e-6, 1e-4, 1e-2, 1.0),
    fit_fraction: float = 0.75,
) -> tuple[torch.Tensor, dict]:
    """Select ridge internally, then construct Down once from all calibration data."""
    if features.ndim != 2 or targets.ndim != 2 or prior.ndim != 2:
        raise ValueError("features, targets, and prior must be matrices")
    retained, tokens = features.shape
    if targets.shape[1] != tokens or prior.shape != (targets.shape[0], retained):
        raise ValueError("readout dimensions do not agree")
    fit_slice, selection_slice = _sequence_split_indices(
        tokens, int(sequence_length), float(fit_fraction),
    )
    hf = features[:, fit_slice]
    yf = targets[:, fit_slice]
    hv = features[:, selection_slice]
    yv = targets[:, selection_slice].float()
    trials = []
    best_ratio = None
    best_error = float("inf")
    for ratio in ridge_ratios:
        prediction = _anchored_validation_prediction(hf, yf, hv, prior, float(ratio))
        error = float(
            torch.linalg.vector_norm(prediction - yv)
            / torch.linalg.vector_norm(yv).clamp_min(1e-30)
        )
        trials.append({"ridge_ratio": float(ratio), "selection_relative_error": error})
        if error < best_error:
            best_error = error
            best_ratio = float(ratio)

    h = features.float()
    y = targets.float()
    b0 = prior.float()
    residual = y - b0 @ h
    gram = h.T @ h
    scale = gram.diagonal().mean().clamp_min(1e-8)
    ridge = best_ratio * scale
    gram.diagonal().add_(ridge + scale * 1e-7)
    # Solve against the output residual in the token-space dual.  This avoids
    # an intractable retained_neurons^2 normal matrix.
    solved = torch.linalg.solve(gram, residual.T)
    readout = b0 + solved.T @ h.T
    metadata = {
        "objective": "calibration_only_prior_anchored_output_ridge",
        "ridge_ratio": best_ratio,
        "ridge_absolute": float(ridge),
        "fit_tokens": int(tokens),
        "selection_fit_tokens": int(hf.shape[1]),
        "selection_validation_tokens": int(hv.shape[1]),
        "trials": trials,
        "construction_uses_validation": False,
    }
    return readout, metadata
