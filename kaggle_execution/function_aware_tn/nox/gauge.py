"""Exact, head-local attention reparameterizations for codec-orbit tests."""

from __future__ import annotations

import torch


@torch.no_grad()
def balanced_attention_gauge(weights, n_heads, n_kv_heads, eps=1e-12):
    """Balance paired attention-channel energies using exact model symmetries.

    For each KV head and RoPE channel pair, Q is scaled by ``s`` while K is
    scaled by ``1/s``. Sharing each scale across the paired coordinates makes
    the diagonal gauge commute with every relative-position RoPE rotation, so
    all cross-position QK scores are preserved. V is scaled by ``t`` and the
    matching input columns of O by ``1/t``, which cancels after attention's
    weighted value sum. GQA query heads sharing one KV head share the same
    paired-channel scales.

    This changes coordinates only. It does not use calibration data, optimize
    task loss, or modify the represented dense block function.
    """
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads:
        raise ValueError("n_heads must be a positive multiple of n_kv_heads")
    required = ("q_proj", "k_proj", "v_proj", "o_proj")
    if any(name not in weights for name in required):
        raise KeyError("weights must contain q_proj, k_proj, v_proj, and o_proj")

    q, k, v, o = (weights[name] for name in required)
    if any(t.ndim != 2 for t in (q, k, v, o)):
        raise ValueError("attention projection weights must be matrices")
    d_head = q.shape[0] // n_heads
    if d_head * n_heads != q.shape[0] or k.shape[0] != n_kv_heads * d_head:
        raise ValueError("Q/K output widths are incompatible with head counts")
    if d_head % 2:
        raise ValueError("RoPE-paired channel balancing requires an even head dimension")
    if v.shape[0] != n_kv_heads * d_head or o.shape[1] != n_heads * d_head:
        raise ValueError("V/O dimensions are incompatible with attention heads")
    if k.shape[1] != q.shape[1] or v.shape[1] != q.shape[1]:
        raise ValueError("Q/K/V input widths must match")

    repeat = n_heads // n_kv_heads
    q_norm = torch.linalg.vector_norm(q.float().reshape(n_heads, d_head, -1), dim=-1)
    k_norm = torch.linalg.vector_norm(k.float().reshape(n_kv_heads, d_head, -1), dim=-1)
    half = d_head // 2
    q_group_energy = q_norm.reshape(n_kv_heads, repeat, d_head).square().mean(dim=1)
    q_pair_norm = ((q_group_energy[:, :half] + q_group_energy[:, half:]) * 0.5).sqrt()
    k_pair_norm = ((k_norm[:, :half].square() + k_norm[:, half:].square()) * 0.5).sqrt()
    qk_pair_scale = torch.sqrt(k_pair_norm.clamp_min(eps) / q_pair_norm.clamp_min(eps))
    qk_scale = torch.cat((qk_pair_scale, qk_pair_scale), dim=1)

    v_norm = torch.linalg.vector_norm(v.float().reshape(n_kv_heads, d_head, -1), dim=-1)
    o_norm = torch.linalg.vector_norm(o.float().reshape(o.shape[0], n_heads, d_head), dim=0)
    o_group_norm = o_norm.reshape(n_kv_heads, repeat, d_head).square().mean(dim=1).sqrt()
    vo_scale = torch.sqrt(o_group_norm.clamp_min(eps) / v_norm.clamp_min(eps))

    if not (torch.isfinite(qk_scale).all() and torch.isfinite(vo_scale).all()):
        raise ValueError("attention weights produced non-finite gauge scales")

    result = dict(weights)
    result["q_proj"] = q * qk_scale.repeat_interleave(repeat, dim=0).reshape(-1, 1).to(q.dtype)
    result["k_proj"] = k / qk_scale.reshape(-1, 1).to(k.dtype)
    result["v_proj"] = v * vo_scale.reshape(-1, 1).to(v.dtype)
    o_scale = vo_scale.repeat_interleave(repeat, dim=0).reshape(1, -1).to(o.dtype)
    result["o_proj"] = o / o_scale

    metadata = {
        "method": "headwise_pairwise_energy_balance_v1",
        "n_heads": int(n_heads),
        "n_kv_heads": int(n_kv_heads),
        "head_dim": int(d_head),
        "qk_scale_min": float(qk_scale.min().item()),
        "qk_scale_max": float(qk_scale.max().item()),
        "vo_scale_min": float(vo_scale.min().item()),
        "vo_scale_max": float(vo_scale.max().item()),
    }
    return result, metadata
