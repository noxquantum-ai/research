"""Small exact coordinate symmetries for Llama-style transformer blocks.

The helpers in this module return transformed projection weights.  Their
mathematical contracts are documented in ``docs/FUNCTION_AWARE_TN_MATHEMATICS.md``.
They do not optimize a task loss or add trainable parameters.
"""

from __future__ import annotations

import math

import torch


def _check_attention(weights, n_heads: int, n_kv_heads: int):
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads:
        raise ValueError("n_heads must be a positive multiple of n_kv_heads")
    names = ("q_proj", "k_proj", "v_proj", "o_proj")
    if any(name not in weights for name in names):
        raise KeyError("weights must contain q_proj, k_proj, v_proj, and o_proj")
    q, k, v, o = (weights[name] for name in names)
    if any(value.ndim != 2 for value in (q, k, v, o)):
        raise ValueError("attention projection weights must be matrices")
    d_head = q.shape[0] // n_heads
    if d_head * n_heads != q.shape[0] or d_head % 2:
        raise ValueError("Q output width must divide into even-sized heads")
    if k.shape[0] != n_kv_heads * d_head or v.shape[0] != n_kv_heads * d_head:
        raise ValueError("K/V output widths do not match the KV head count")
    if o.shape[1] != n_heads * d_head:
        raise ValueError("O input width does not match the query head count")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("Q/K/V input widths must match")
    return q, k, v, o, d_head


def _rope_pair_action(rows: torch.Tensor, scales: torch.Tensor, angles: torch.Tensor):
    """Apply per-pair ``s R(theta)`` in Llama's split-half RoPE layout."""
    half = rows.shape[-2] // 2
    first, second = rows[..., :half, :], rows[..., half:, :]
    scale = scales[..., None]
    cosine = angles.cos()[..., None]
    sine = angles.sin()[..., None]
    mapped_first = scale * (cosine * first - sine * second)
    mapped_second = scale * (sine * first + cosine * second)
    return torch.cat((mapped_first, mapped_second), dim=-2)


def rope_commuting_qk_gauge(
    weights: dict[str, torch.Tensor],
    n_heads: int,
    n_kv_heads: int,
    *,
    pair_scales: torch.Tensor | None = None,
    pair_angles: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply a head-local Q/K gauge from the commutant of Llama RoPE.

    For each rotary pair, the map is ``A=s R(theta)`` on Q and ``A^-T`` on K.
    ``s`` and ``theta`` are shared across GQA query heads attached to a KV head.
    This is exact because such block maps commute with every relative RoPE
    rotation.  Arbitrary head rotations or arbitrary channel mixing are not
    accepted by this helper.
    """
    q, k, _, _, d_head = _check_attention(weights, n_heads, n_kv_heads)
    device = q.device
    dtype = torch.float32
    half = d_head // 2
    expected = (n_kv_heads, half)
    if pair_scales is None:
        pair_scales = torch.ones(expected, device=device, dtype=dtype)
    else:
        pair_scales = torch.as_tensor(pair_scales, device=device, dtype=dtype)
    if pair_angles is None:
        pair_angles = torch.zeros(expected, device=device, dtype=dtype)
    else:
        pair_angles = torch.as_tensor(pair_angles, device=device, dtype=dtype)
    if pair_scales.shape != expected or pair_angles.shape != expected:
        raise ValueError(f"pair scales and angles must have shape {expected}")
    if not torch.isfinite(pair_scales).all() or torch.any(pair_scales <= 0):
        raise ValueError("pair scales must be finite and positive")
    if not torch.isfinite(pair_angles).all():
        raise ValueError("pair angles must be finite")

    repeat = n_heads // n_kv_heads
    q_view = q.float().reshape(n_heads, d_head, q.shape[1])
    k_view = k.float().reshape(n_kv_heads, d_head, k.shape[1])
    q_scale = pair_scales.repeat_interleave(repeat, dim=0)
    q_angle = pair_angles.repeat_interleave(repeat, dim=0)
    result = dict(weights)
    result["q_proj"] = _rope_pair_action(q_view, q_scale, q_angle).reshape_as(q).to(q.dtype)
    result["k_proj"] = _rope_pair_action(
        k_view, pair_scales.reciprocal(), pair_angles,
    ).reshape_as(k).to(k.dtype)
    return result


def _contiguous_pair_action(rows: torch.Tensor, scales: torch.Tensor, angles: torch.Tensor):
    """Apply independent two-channel ``s R(theta)`` maps along a head."""
    if rows.shape[-2] % 2:
        raise ValueError("head dimension must be even for pairwise factors")
    pairs = rows.reshape(*rows.shape[:-2], rows.shape[-2] // 2, 2, rows.shape[-1])
    x0, x1 = pairs[..., 0, :], pairs[..., 1, :]
    scale = scales[..., None]
    c = angles.cos()[..., None]
    s = angles.sin()[..., None]
    y0 = scale * (c * x0 - s * x1)
    y1 = scale * (s * x0 + c * x1)
    return torch.stack((y0, y1), dim=-2).reshape_as(rows)


def value_output_gauge(
    weights: dict[str, torch.Tensor],
    n_heads: int,
    n_kv_heads: int,
    *,
    pair_scales: torch.Tensor | None = None,
    pair_angles: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply a per-head invertible V/O change of coordinates.

    Each V head is transformed by ``A=s R(theta)`` and the matching columns
    of O by ``A^-1``.  GQA repeats the inverse map for every query head using
    that KV head.  There is no RoPE on V, so contiguous channel pairs may be
    mixed independently.
    """
    _, _, v, o, d_head = _check_attention(weights, n_heads, n_kv_heads)
    device = v.device
    expected = (n_kv_heads, d_head // 2)
    scales = torch.ones(expected, device=device, dtype=torch.float32) if pair_scales is None else torch.as_tensor(pair_scales, device=device, dtype=torch.float32)
    angles = torch.zeros(expected, device=device, dtype=torch.float32) if pair_angles is None else torch.as_tensor(pair_angles, device=device, dtype=torch.float32)
    if scales.shape != expected or angles.shape != expected:
        raise ValueError(f"pair scales and angles must have shape {expected}")
    if not torch.isfinite(scales).all() or torch.any(scales <= 0) or not torch.isfinite(angles).all():
        raise ValueError("V/O gauge factors must be finite with positive scales")

    v_view = v.float().reshape(n_kv_heads, d_head, v.shape[1])
    v_new = _contiguous_pair_action(v_view, scales, angles).reshape_as(v)
    inverse_scales = scales.reciprocal()
    inverse_angles = -angles
    repeat = n_heads // n_kv_heads
    o_blocks = o.float().reshape(o.shape[0], n_kv_heads, repeat, d_head)
    o_pairs = o_blocks.reshape(o.shape[0], n_kv_heads, repeat, d_head // 2, 2)
    x0, x1 = o_pairs[..., 0], o_pairs[..., 1]
    scale = inverse_scales[None, :, None, :]
    c = inverse_angles.cos()[None, :, None, :]
    s = inverse_angles.sin()[None, :, None, :]
    # Right multiplication by A^-1 for row vectors of O coefficients.
    y0 = scale * (c * x0 + s * x1)
    y1 = scale * (-s * x0 + c * x1)
    o_new = torch.stack((y0, y1), dim=-1).reshape_as(o)
    result = dict(weights)
    result["v_proj"] = v_new.to(v.dtype)
    result["o_proj"] = o_new.to(o.dtype)
    return result


def swiglu_neuron_permutation_gauge(
    weights: dict[str, torch.Tensor], permutation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Permute the shared SwiGLU neuron axis in Gate, Up, and Down."""
    gate, up, down = (weights[name] for name in ("gate_proj", "up_proj", "down_proj"))
    if gate.ndim != 2 or up.shape != gate.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError("incompatible SwiGLU projection shapes")
    permutation = torch.as_tensor(permutation, dtype=torch.long, device=gate.device)
    width = gate.shape[0]
    if permutation.shape != (width,) or not torch.equal(
        torch.sort(permutation).values, torch.arange(width, device=gate.device),
    ):
        raise ValueError("permutation must contain every SwiGLU neuron once")
    result = dict(weights)
    result["gate_proj"] = gate.index_select(0, permutation)
    result["up_proj"] = up.index_select(0, permutation)
    result["down_proj"] = down.index_select(1, permutation)
    return result


def swiglu_up_down_scaling_gauge(
    weights: dict[str, torch.Tensor], scales: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Apply the exact reciprocal diagonal gauge to Up and Down."""
    gate, up, down = (weights[name] for name in ("gate_proj", "up_proj", "down_proj"))
    if gate.ndim != 2 or up.shape != gate.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError("incompatible SwiGLU projection shapes")
    scales = torch.as_tensor(scales, device=up.device, dtype=up.dtype)
    if scales.shape != (up.shape[0],) or not torch.isfinite(scales).all() or torch.any(scales == 0):
        raise ValueError("scales must be finite, nonzero, and match the neuron width")
    result = dict(weights)
    result["up_proj"] = up * scales[:, None]
    result["down_proj"] = down / scales[None, :]
    return result


def signed_permutation_residual_gauge(
    weights: dict[str, torch.Tensor],
    permutation: torch.Tensor,
    signs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Change the residual-stream basis by a signed permutation.

    The same permutation must be used for all blocks and the embedding/head
    when transforming a complete model.  Projection names follow HuggingFace
    Llama linear-weight layout.
    """
    permutation = torch.as_tensor(permutation, dtype=torch.long)
    d_model = int(permutation.numel())
    if not torch.equal(torch.sort(permutation).values, torch.arange(d_model)):
        raise ValueError("permutation must contain every residual coordinate once")
    if signs is None:
        signs = torch.ones(d_model, dtype=torch.float32)
    signs = torch.as_tensor(signs, dtype=torch.float32)
    if signs.shape != (d_model,) or not torch.all((signs == 1) | (signs == -1)):
        raise ValueError("signs must contain one +1 or -1 per residual coordinate")
    result = dict(weights)
    input_names = (
        "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
        "embed_tokens", "lm_head",
    )
    output_names = ("o_proj", "down_proj")
    for name in input_names:
        if name in weights:
            w = weights[name]
            if w.shape[1] != d_model:
                raise ValueError(f"{name} has incompatible residual input width")
            p = permutation.to(device=w.device)
            s = signs.to(device=w.device, dtype=w.dtype)
            result[name] = w.index_select(1, p) * s.unsqueeze(0)
    for name in output_names:
        if name in weights:
            w = weights[name]
            if w.shape[0] != d_model:
                raise ValueError(f"{name} has incompatible residual output width")
            p = permutation.to(device=w.device)
            s = signs.to(device=w.device, dtype=w.dtype)
            result[name] = w.index_select(0, p) * s.unsqueeze(1)
    for name in ("input_layernorm", "post_attention_layernorm", "final_norm"):
        if name in weights:
            gamma = weights[name]
            if gamma.shape != (d_model,):
                raise ValueError(f"{name} has incompatible residual width")
            result[name] = gamma.index_select(0, permutation.to(gamma.device))
    return result


def rope_rotation_frequencies(d_head: int, *, base: float = 10000.0, device=None):
    """Return inverse frequencies for the legacy Llama-2 split-half RoPE."""
    if d_head <= 0 or d_head % 2:
        raise ValueError("RoPE head dimension must be a positive even number")
    return 1.0 / (base ** (torch.arange(0, d_head, 2, device=device, dtype=torch.float32) / d_head))


def _rotate_half(x: torch.Tensor):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor):
    # x is [batch, heads, sequence, head_dim].
    frequencies = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return x * angles.cos()[None, None, :, :] + _rotate_half(x) * angles.sin()[None, None, :, :]


def llama_attention_output(
    weights: dict[str, torch.Tensor],
    inputs: torch.Tensor,
    *,
    n_heads: int,
    n_kv_heads: int,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """Small eager Llama attention reference used by exactness tests.

    ``inputs`` is ``[batch, sequence, hidden]``.  The helper uses causal softmax
    and grouped-query head repetition, and intentionally omits dropout/cache.
    """
    q_w, k_w, v_w, o_w, d_head = _check_attention(weights, n_heads, n_kv_heads)
    batch, seq, hidden = inputs.shape
    q = torch.nn.functional.linear(inputs, q_w).reshape(batch, seq, n_heads, d_head).transpose(1, 2)
    k = torch.nn.functional.linear(inputs, k_w).reshape(batch, seq, n_kv_heads, d_head).transpose(1, 2)
    v = torch.nn.functional.linear(inputs, v_w).reshape(batch, seq, n_kv_heads, d_head).transpose(1, 2)
    inv = rope_rotation_frequencies(d_head, base=rope_base, device=inputs.device)
    positions = torch.arange(seq, device=inputs.device)
    q = _apply_rope(q, positions, inv)
    k = _apply_rope(k, positions, inv)
    repeat = n_heads // n_kv_heads
    k = k.repeat_interleave(repeat, dim=1)
    v = v.repeat_interleave(repeat, dim=1)
    logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(d_head)
    causal = torch.ones((seq, seq), device=inputs.device, dtype=torch.bool).triu(1)
    logits = logits.masked_fill(causal, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(logits, dim=-1).to(v.dtype)
    attended = torch.matmul(probabilities, v).transpose(1, 2).reshape(batch, seq, n_heads * d_head)
    return torch.nn.functional.linear(attended, o_w)
