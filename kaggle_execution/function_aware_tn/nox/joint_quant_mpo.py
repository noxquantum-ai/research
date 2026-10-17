"""Joint low-bit and tensor-network operator construction.

The codec alternates between a packed groupwise base and a packed MPO which
models the *decoded* base codec's remaining error.  All fitting happens before
the artifact is finalized; evaluation and reconstruction are read-only.
"""

from __future__ import annotations

import math

import torch

from nox.packing import pack_factor, tensor_payload_bytes, unpack_factor
from nox.residual_codec import decode_groupwise, encode_groupwise
from nox.utils import get_modes_for_projection, mpo_to_dense


CODEC_TYPE = "joint_quant_mpo_v1"


def _low_rank_svd(matrix: torch.Tensor, rank: int, *, oversample: int, niter: int,
                  generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic randomized truncated SVD without full square factors."""
    rows, columns = matrix.shape
    rank = min(int(rank), rows, columns)
    if rank <= 0:
        raise ValueError("rank must be positive")
    sample_rank = min(rank + int(oversample), rows, columns)
    # Work on the smaller left dimension to limit the random sketch and QR.
    transposed = rows > columns
    source = matrix.T if transposed else matrix
    omega = torch.randn(
        source.shape[1], sample_rank, device=source.device, dtype=source.dtype,
        generator=generator,
    )
    sample = source @ omega
    for _ in range(int(niter)):
        sample = source @ (source.T @ sample)
    basis = torch.linalg.qr(sample, mode="reduced").Q
    reduced = basis.T @ source
    u_small, singular_values, vh = torch.linalg.svd(reduced, full_matrices=False)
    u = basis @ u_small
    u, singular_values, vh = u[:, :rank], singular_values[:rank], vh[:rank]
    if transposed:
        return vh.T.contiguous(), singular_values, u.T.contiguous()
    return u.contiguous(), singular_values, vh.contiguous()


def randomized_tt_svd_4site(
    matrix: torch.Tensor,
    o_modes: tuple[int, int, int, int],
    i_modes: tuple[int, int, int, int],
    ranks: tuple[int, int, int, int, int],
    *,
    oversample: int = 8,
    niter: int = 1,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Approximate a matrix by a four-site MPO at fixed bond ranks."""
    if math.prod(o_modes) != matrix.shape[0] or math.prod(i_modes) != matrix.shape[1]:
        raise ValueError("mode products must match matrix shape")
    if len(ranks) != 5 or ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("ranks must have form (1, r1, r2, r3, 1)")
    generator = torch.Generator(device=matrix.device)
    generator.manual_seed(int(seed))
    tensor = matrix.reshape(*o_modes, *i_modes)
    remainder = tensor.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    cores: list[torch.Tensor] = []
    previous_rank = 1
    for site in range(3):
        unfolding = remainder.reshape(previous_rank * o_modes[site] * i_modes[site], -1)
        u, singular_values, vh = _low_rank_svd(
            unfolding, ranks[site + 1], oversample=oversample, niter=niter,
            generator=generator,
        )
        next_rank = singular_values.numel()
        cores.append(u.reshape(previous_rank, o_modes[site], i_modes[site], next_rank))
        remainder = singular_values.unsqueeze(1) * vh
        previous_rank = next_rank
    cores.append(remainder.reshape(previous_rank, o_modes[3], i_modes[3], 1))
    return cores


def _decode_correction(packed_cores: list[dict], column_scale: torch.Tensor | None,
                       *, device: torch.device) -> torch.Tensor:
    cores = [unpack_factor(core).to(device=device, dtype=torch.float32) for core in packed_cores]
    correction = mpo_to_dense(cores)
    if column_scale is not None:
        correction = correction / column_scale.to(device=device, dtype=torch.float32).unsqueeze(0)
    return correction


def decode_joint_quant_mpo(artifact: dict, *, device="cpu", dtype=torch.float32) -> torch.Tensor:
    if artifact.get("codec_type") != CODEC_TYPE:
        raise ValueError(f"unsupported codec type: {artifact.get('codec_type')!r}")
    target_device = torch.device(device)
    base = decode_groupwise(artifact["base_codec"], device=target_device, dtype=torch.float32)
    scale = artifact["base_codec"].get("column_scales")
    correction = _decode_correction(artifact["correction_cores"], scale, device=target_device)
    if tuple(base.shape) != tuple(correction.shape):
        raise ValueError("base and MPO correction shapes differ")
    return (base + correction).to(dtype=dtype)


def joint_quant_mpo_payload_bytes(artifact: dict) -> int:
    return int(tensor_payload_bytes(artifact["base_codec"]) + tensor_payload_bytes(artifact["correction_cores"]))


@torch.no_grad()
def encode_joint_quant_mpo(
    matrix: torch.Tensor,
    calibration_context: torch.Tensor,
    *,
    bits: int = 3,
    group_size: int = 64,
    alpha: float = 0.5,
    ranks: tuple[int, int, int, int, int] = (1, 16, 32, 16, 1),
    passes: int = 2,
    core_group_size: int = 64,
    oversample: int = 8,
    power_iterations: int = 1,
    seed: int = 0,
) -> dict:
    """Alternately construct a low-bit base and complementary packed MPO.

    The selection criterion is the exact decoded calibration objective
    ``||(W - W_hat) X||_F^2 / ||W X||_F^2``.  TT-SVD uses the activation-RMS
    diagonal as a cheap covariance surrogate, then a closed-form scalar refit
    aligns the MPO with the true functional residual before core packing.
    """
    if matrix.ndim != 2 or calibration_context.ndim != 2:
        raise ValueError("matrix and calibration_context must both be rank two")
    if matrix.shape[1] != calibration_context.shape[0]:
        raise ValueError("calibration context width does not match matrix")
    if passes <= 0:
        raise ValueError("passes must be positive")
    device = matrix.device
    source = matrix.detach().to(dtype=torch.float32)
    context = calibration_context.detach().to(device=device, dtype=torch.float32)
    activation_rms = torch.sqrt(torch.mean(context.square(), dim=1))
    o_modes, i_modes = get_modes_for_projection(*source.shape)
    target_output = source @ context
    target_energy = target_output.square().sum().clamp_min(1e-20)
    correction = torch.zeros_like(source)
    best_artifact = None
    best_loss = float("inf")
    pass_metrics = []

    for pass_index in range(int(passes)):
        base_codec = encode_groupwise(
            source - correction, bits=bits, group_size=group_size,
            activation_rms=activation_rms, alpha=alpha,
        )
        base = decode_groupwise(base_codec, device=device, dtype=torch.float32)
        residual = source - base
        column_scale = base_codec.get("column_scales")
        weighted_residual = residual if column_scale is None else (
            residual * column_scale.to(device=device, dtype=torch.float32).unsqueeze(0)
        )
        raw_cores = randomized_tt_svd_4site(
            weighted_residual, o_modes, i_modes, ranks,
            oversample=oversample, niter=power_iterations,
            seed=int(seed) + pass_index,
        )
        raw_correction = mpo_to_dense(raw_cores)
        if column_scale is not None:
            raw_correction = raw_correction / column_scale.to(device=device, dtype=torch.float32).unsqueeze(0)
        residual_output = residual @ context
        correction_output = raw_correction @ context
        gain = (residual_output * correction_output).sum() / correction_output.square().sum().clamp_min(1e-20)
        raw_cores[0] = raw_cores[0] * gain
        packed_cores = [pack_factor(core, group_size=core_group_size) for core in raw_cores]
        correction = _decode_correction(packed_cores, column_scale, device=device)
        reconstructed = base + correction
        relative_loss = float(((target_output - reconstructed @ context).square().sum() / target_energy).item())
        pass_metrics.append({
            "pass": pass_index + 1,
            "decoded_calibration_relative_squared_error": relative_loss,
            "functional_gain_before_core_packing": float(gain),
        })
        if relative_loss < best_loss:
            best_loss = relative_loss
            best_artifact = {
                "codec_type": CODEC_TYPE,
                "codec_kind": "joint_quant_mpo",
                "base_codec": base_codec,
                "correction_cores": packed_cores,
                "orig_shape": tuple(source.shape),
                "o_modes": tuple(o_modes),
                "i_modes": tuple(i_modes),
                "ranks": tuple(int(value) for value in ranks),
                "construction_pass": pass_index + 1,
                "construction_objective": "decoded_calibration_functional_squared_error",
                "gradient_healing": False,
            }

    assert best_artifact is not None
    best_artifact["pass_metrics"] = pass_metrics
    best_artifact["decoded_calibration_relative_squared_error"] = best_loss
    best_artifact["payload_bytes"] = joint_quant_mpo_payload_bytes(best_artifact)
    return best_artifact
