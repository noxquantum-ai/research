"""Quantization-aware helpers for an exploratory post-compression healing diagnostic."""

import torch
import torch.nn.functional as F

from nox.compiler import quantize_4bit_grouped
from nox.utils import mpo_to_dense
from nox.block_compiler import scaled_dot_product_attention_2d


def post_compression_refinement_metadata():
    """Return the classification for optimizer updates to a saved checkpoint."""
    return {
        "healing": True,
        "post_compression_factor_updates": True,
        "diagnostic_only": True,
    }


def fake_quantize_ste(tensor, group_size=64):
    """Use the existing grouped-int4 forward values with a straight-through gradient."""
    quantized = quantize_4bit_grouped(tensor, group_size)
    return tensor + (quantized - tensor).detach()


def make_trainable_components(artifact, device):
    """Create FP32 master parameters for the artifact's quantized factors."""
    if artifact.get("quantization_bits", 4) != 4:
        raise ValueError("Block distillation expects simulated-int4 Nox factors")
    result = {
        "cores": [torch.nn.Parameter(core.to(device=device, dtype=torch.float32).clone())
                  for core in artifact["cores"]],
        "A_bridge": None,
        "B_bridge": None,
    }
    for name in ("A_bridge", "B_bridge"):
        if artifact.get(name) is not None:
            result[name] = torch.nn.Parameter(
                artifact[name].to(device=device, dtype=torch.float32).clone()
            )
    return result


def reconstruct_trainable_artifact(artifact, components, device):
    """Reconstruct a dense matrix while differentiating through int4 factors."""
    cores = [fake_quantize_ste(core) for core in components["cores"]]
    d_awt = artifact["d_awt"].to(device=device, dtype=torch.float32)
    weight = mpo_to_dense(cores) / d_awt.unsqueeze(0)
    if components.get("A_bridge") is not None and components.get("B_bridge") is not None:
        a_bridge = fake_quantize_ste(components["A_bridge"])
        b_bridge = fake_quantize_ste(components["B_bridge"])
        weight = weight + a_bridge @ b_bridge
    indices = artifact.get("sparse_indices")
    values = artifact.get("sparse_values")
    if indices is not None and values is not None:
        indices = indices.to(device=device, dtype=torch.long)
        flat_indices = indices[:, 0] * weight.shape[1] + indices[:, 1]
        sparse = torch.zeros(weight.numel(), device=device, dtype=weight.dtype)
        sparse = sparse.scatter_add(0, flat_indices, values.to(device=device, dtype=weight.dtype))
        weight = weight + sparse.reshape_as(weight)
    return weight


def forward_block_with_grad(weights, x, n_heads=32, n_kv_heads=32, seq_len=128, rms_norm_eps=1e-5):
    """Differentiable equivalent of the repository's LLaMA block forward path."""
    def rms_norm(value, scale):
        value = value * torch.rsqrt(value.pow(2).mean(dim=0, keepdim=True) + rms_norm_eps)
        return value * scale.unsqueeze(1)

    x_norm = rms_norm(x, weights["input_layernorm"])
    q = weights["q_proj"] @ x_norm
    k = weights["k_proj"] @ x_norm
    v = weights["v_proj"] @ x_norm
    attention = scaled_dot_product_attention_2d(q, k, v, n_heads, n_kv_heads, seq_len)
    residual = x + weights["o_proj"] @ attention
    mlp_input = rms_norm(residual, weights["post_attention_layernorm"])
    gate = weights["gate_proj"] @ mlp_input
    up = weights["up_proj"] @ mlp_input
    mlp = weights["down_proj"] @ (F.silu(gate) * up)
    return residual + mlp


def relative_output_error(reference, candidate):
    return (candidate.float() - reference.float()).norm() / reference.float().norm().clamp_min(1e-12)
