"""Activation-conditioned Schmidt corrections for an existing NEC operator.

The correction is a low-rank operator written in its Schmidt form ``L @ R``.
Its right Schmidt subspace is obtained from the activations that the compiled
prefix actually produces.  The left factor is then solved in closed form
against a clean teacher target.  This keeps the construction gradient free
while allowing a downstream NEC operator to compensate for upstream damage.
"""

from __future__ import annotations

import torch


CODEC_TYPE = "nec_activation_schmidt_correction_v1"


def _tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def reconstruct_schmidt_correction(
    artifact,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
):
    """Decode a base NEC operator plus its activation-conditioned correction."""

    from nox.compiler import reconstruct_from_artifact

    if artifact.get("codec_type") != CODEC_TYPE:
        raise ValueError(f"unsupported Schmidt correction: {artifact.get('codec_type')!r}")
    base = reconstruct_from_artifact(artifact["base_artifact"], device=device, dtype=dtype)
    left = artifact["left_factor"].to(device=device, dtype=dtype)
    right = artifact["right_factor"].to(device=device, dtype=dtype)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("malformed Schmidt correction factors")
    if base.shape != (left.shape[0], right.shape[1]):
        raise ValueError("base and Schmidt correction shapes differ")
    return base.addmm(left, right)


@torch.no_grad()
def compile_schmidt_correction(
    base_artifact,
    inputs: torch.Tensor,
    teacher_outputs: torch.Tensor,
    *,
    rank: int,
    ridge_ratio: float = 0.02,
    oversample: int = 16,
    power_iterations: int = 1,
    seed: int = 0,
):
    """Fit a byte-explicit Schmidt correction on corrupted-prefix inputs.

    For ``X = U diag(s) V^T``, the ridge solution restricted to the first
    ``rank`` activation Schmidt coordinates is

    ``L = (Y - W_base X) V diag(s / (s^2 + lambda))`` and ``R = U^T``.

    The returned factors are FP16 serialized tensors.  The fit itself and all
    diagnostics are evaluated in FP32.
    """

    from nox.compiler import reconstruct_from_artifact

    if inputs.ndim != 2 or teacher_outputs.ndim != 2:
        raise ValueError("inputs and teacher outputs must be matrices")
    if inputs.shape[1] != teacher_outputs.shape[1]:
        raise ValueError("input and teacher token counts differ")
    if rank <= 0 or ridge_ratio <= 0 or oversample < 0 or power_iterations < 0:
        raise ValueError("invalid Schmidt fit configuration")
    if not torch.isfinite(inputs).all() or not torch.isfinite(teacher_outputs).all():
        raise ValueError("Schmidt fit received non-finite calibration values")

    device = inputs.device
    x = inputs.float()
    target = teacher_outputs.float()
    base = reconstruct_from_artifact(base_artifact, device=device, dtype=torch.float32)
    if base.shape[1] != x.shape[0] or base.shape[0] != target.shape[0]:
        raise ValueError("base operator is incompatible with calibration matrices")
    residual = target - base @ x

    actual_rank = min(int(rank), x.shape[0], x.shape[1])
    sketch_rank = min(actual_rank + int(oversample), x.shape[0], x.shape[1])
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    # torch.svd_lowrank has no generator argument. Forking the RNG makes this
    # construction repeatable without changing the caller's random stream.
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        left_basis, singular_values, sample_basis = torch.svd_lowrank(
            x, q=sketch_rank, niter=int(power_iterations),
        )
    left_basis = left_basis[:, :actual_rank]
    singular_values = singular_values[:actual_rank]
    sample_basis = sample_basis[:, :actual_rank]
    ridge = (float(ridge_ratio) * singular_values[0].float()).square().clamp_min(1e-12)
    response = residual @ sample_basis
    response.mul_((singular_values / (singular_values.square() + ridge))[None, :])
    right = left_basis.T.contiguous()
    correction_output = response @ (right @ x)
    corrected = base @ x + correction_output

    target_energy = target.square().sum().clamp_min(1e-30)
    before = float((target - base @ x).square().sum() / target_energy)
    after = float((target - corrected).square().sum() / target_energy)
    artifact = {
        "codec_type": CODEC_TYPE,
        "base_artifact": base_artifact,
        "left_factor": response.to(device="cpu", dtype=torch.float16),
        "right_factor": right.to(device="cpu", dtype=torch.float16),
        "orig_shape": (int(base.shape[0]), int(base.shape[1])),
        "rank": actual_rank,
        "construction": "corrupted_activation_schmidt_ridge",
        "ridge_ratio": float(ridge_ratio),
        "ridge": float(ridge),
        "calibration_tokens": int(x.shape[1]),
        "relative_output_error_before": before ** 0.5,
        "relative_output_error_after": after ** 0.5,
        "gradient_healing": False,
    }
    artifact["payload_bytes"] = _tensor_bytes(artifact)
    return artifact, corrected

