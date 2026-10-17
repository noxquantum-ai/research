"""Deterministic packed low-bit groupwise codecs for matrix weights.

Codes are signed symmetric integers stored as tightly packed unsigned bit
fields.  Every group is a contiguous, row-local run of input columns.  Scale
tensors are kept in FP16 and included in :func:`codec_payload_bytes`.
"""

from __future__ import annotations

import math

import torch


SUPPORTED_BITS = (2, 3, 4)
CODEC_TYPE = "groupwise_packed_v1"


def _validate_bits(bits: int) -> None:
    if int(bits) not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack 0 <= codes < 2**bits into a dense uint8 bitstream."""
    codes = codes.reshape(-1).to(dtype=torch.int64)
    if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= (1 << bits)):
        raise ValueError("code lies outside the selected unsigned bit width")
    # Use int32 accumulation: CUDA scatter-add has reliable support for this
    # dtype, and individual bit contributions never overlap.
    packed = torch.zeros((codes.numel() * bits + 7) // 8, dtype=torch.int32, device=codes.device)
    bit_offsets = torch.arange(codes.numel(), device=codes.device, dtype=torch.int64) * bits
    for bit in range(bits):
        positions = bit_offsets + bit
        byte_indices = torch.div(positions, 8, rounding_mode="floor")
        bit_indices = positions.remainder(8)
        values = ((codes >> bit) & 1) << bit_indices
        packed.scatter_add_(0, byte_indices, values.to(torch.int32))
    return packed.to(torch.uint8)


def _unpack_codes(packed: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    """Unpack the first ``count`` low-bit fields from a uint8 stream."""
    packed = packed.reshape(-1).to(dtype=torch.uint8)
    if packed.numel() * 8 < int(count) * bits:
        raise ValueError("packed code buffer is shorter than its declared shape")
    positions = torch.arange(int(count), device=packed.device, dtype=torch.int64) * bits
    result = torch.zeros(int(count), dtype=torch.int16, device=packed.device)
    for bit in range(bits):
        bit_positions = positions + bit
        byte_indices = torch.div(bit_positions, 8, rounding_mode="floor")
        bit_indices = bit_positions.remainder(8)
        values = (packed[byte_indices].to(torch.int16) >> bit_indices) & 1
        result |= values << bit
    return result


def encode_groupwise(
    matrix: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
    activation_rms: torch.Tensor | None = None,
    alpha: float = 0.0,
    min_scale: float = 0.1,
    max_scale: float = 10.0,
) -> dict:
    """Encode a finite ``[out_features, in_features]`` matrix.

    With activation scaling, the encoder forms ``s_j = clamp(rms_j**alpha)``
    from calibration activations, quantizes ``W[:, j] * s_j``, and the decoder
    divides by the stored FP16 ``s_j``.  Statistics must be supplied by the
    caller; this function never inspects validation or test examples.
    """
    _validate_bits(bits)
    if not torch.is_tensor(matrix) or matrix.ndim != 2 or not matrix.is_floating_point():
        raise ValueError("matrix must be a floating-point [rows, columns] tensor")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError("matrix dimensions must be nonzero")
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")
    if not math.isfinite(float(alpha)) or float(alpha) < 0:
        raise ValueError("alpha must be finite and nonnegative")
    if not (math.isfinite(float(min_scale)) and math.isfinite(float(max_scale))):
        raise ValueError("scale bounds must be finite")
    if min_scale <= 0 or max_scale < min_scale:
        raise ValueError("require 0 < min_scale <= max_scale")

    source = matrix.detach().to(dtype=torch.float32)
    if not torch.isfinite(source).all():
        raise ValueError("matrix contains NaN or infinite values")
    rows, columns = map(int, source.shape)
    column_scale = None
    if activation_rms is not None:
        rms = torch.as_tensor(activation_rms, device=source.device, dtype=torch.float32).reshape(-1)
        if rms.numel() != columns:
            raise ValueError(f"activation_rms must contain {columns} channel values")
        if not torch.isfinite(rms).all() or torch.any(rms < 0):
            raise ValueError("activation_rms must contain finite nonnegative values")
        if alpha != 0:
            # Zero-activity channels remain finite through the declared floor.
            column_scale_fp32 = rms.clamp_min(torch.finfo(torch.float32).tiny).pow(float(alpha))
            column_scale_fp32 = column_scale_fp32.clamp(float(min_scale), float(max_scale))
            column_scale = column_scale_fp32.to(torch.float16)
            source = source * column_scale_fp32.unsqueeze(0)

    groups_per_row = (columns + int(group_size) - 1) // int(group_size)
    padded_columns = groups_per_row * int(group_size)
    if padded_columns != columns:
        source = torch.nn.functional.pad(source, (0, padded_columns - columns))
    blocks = source.reshape(rows, groups_per_row, int(group_size))

    qmax = (1 << (int(bits) - 1)) - 1
    qmin = -qmax
    scale = blocks.abs().amax(dim=-1) / float(qmax)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(blocks / safe_scale.unsqueeze(-1)).clamp(qmin, qmax).to(torch.int16)
    codes = (quantized - qmin).to(torch.uint8)

    artifact = {
        "codec_type": CODEC_TYPE,
        "shape": (rows, columns),
        "bits": int(bits),
        "group_size": int(group_size),
        "groups_per_row": int(groups_per_row),
        "qmin": int(qmin),
        "qmax": int(qmax),
        "packed_codes": _pack_codes(codes, int(bits)).detach().cpu(),
        "group_scales": scale.to(torch.float16).detach().cpu(),
        "column_scales": None if column_scale is None else column_scale.detach().cpu(),
        "rounding": "torch_round_ties_to_even",
        "scale_dtype": "float16",
    }
    artifact["payload_bytes"] = codec_payload_bytes(artifact)
    return artifact


def decode_groupwise(artifact: dict, *, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Decode a packed groupwise matrix into the requested device and dtype."""
    if artifact.get("codec_type") != CODEC_TYPE:
        raise ValueError(f"unsupported codec type: {artifact.get('codec_type')!r}")
    bits = int(artifact["bits"])
    _validate_bits(bits)
    rows, columns = map(int, artifact["shape"])
    group_size = int(artifact["group_size"])
    groups_per_row = int(artifact["groups_per_row"])
    count = rows * groups_per_row * group_size
    codes = _unpack_codes(artifact["packed_codes"].to(device=device), count, bits)
    qmin = int(artifact["qmin"])
    quantized = (codes + qmin).reshape(rows, groups_per_row, group_size).to(torch.float32)
    scales = artifact["group_scales"].to(device=device, dtype=torch.float32)
    if tuple(scales.shape) != (rows, groups_per_row):
        raise ValueError("group scale shape does not match codec metadata")
    reconstructed = quantized * scales.unsqueeze(-1)
    reconstructed = reconstructed.reshape(rows, groups_per_row * group_size)[:, :columns]
    column_scales = artifact.get("column_scales")
    if column_scales is not None:
        scales_by_column = column_scales.to(device=device, dtype=torch.float32).reshape(1, -1)
        if scales_by_column.shape[1] != columns or torch.any(scales_by_column <= 0):
            raise ValueError("invalid activation scale vector")
        reconstructed = reconstructed / scales_by_column
    if not torch.isfinite(reconstructed).all():
        raise ValueError("decoded matrix contains NaN or infinite values")
    return reconstructed.to(dtype=dtype)


def codec_payload_bytes(artifact: dict) -> int:
    """Exact byte count of codec tensor payloads (container metadata excluded)."""
    names = ("packed_codes", "group_scales", "column_scales")
    return sum(
        int(t.numel() * t.element_size())
        for name in names
        if torch.is_tensor(t := artifact.get(name))
    )


def make_artifact(codec: dict, *, kind: str, **metadata) -> dict:
    """Attach a public candidate kind while retaining the codec schema."""
    if kind not in {"direct_quant", "nec_hybrid_residual"}:
        raise ValueError(f"unsupported candidate kind: {kind}")
    return {**codec, "codec_kind": kind, **metadata}
