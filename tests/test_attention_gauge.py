import unittest

import torch

from nox.gauge import balanced_attention_gauge
from nox.block_compiler import apply_llama2_rope, compute_llama2_rotary_emb
from nox.sequential_compiler import forward_transformer_block


class AttentionGaugeTests(unittest.TestCase):
    def _weights(self, n_heads, n_kv_heads, d_model=16, d_ff=24):
        d_head = d_model // n_heads
        torch.manual_seed(73)
        return {
            "q_proj": torch.randn(d_model, d_model) * 0.2,
            "k_proj": torch.randn(n_kv_heads * d_head, d_model) * 0.2,
            "v_proj": torch.randn(n_kv_heads * d_head, d_model) * 0.2,
            "o_proj": torch.randn(d_model, d_model) * 0.2,
            "gate_proj": torch.randn(d_ff, d_model) * 0.2,
            "up_proj": torch.randn(d_ff, d_model) * 0.2,
            "down_proj": torch.randn(d_model, d_ff) * 0.2,
            "input_layernorm": torch.ones(d_model),
            "post_attention_layernorm": torch.ones(d_model),
        }

    def test_gauge_preserves_dense_block_for_mha_and_gqa_with_rope(self):
        for n_heads, n_kv_heads in ((4, 4), (4, 2)):
            weights = self._weights(n_heads, n_kv_heads)
            gauged, metadata = balanced_attention_gauge(weights, n_heads, n_kv_heads)
            x = torch.randn(16, 16) * 0.1
            d_head = 16 // n_heads
            q = (weights["q_proj"] @ x).reshape(n_heads, d_head, 2, 8).permute(2, 0, 3, 1)
            k = (weights["k_proj"] @ x).reshape(n_kv_heads, d_head, 2, 8).permute(2, 0, 3, 1)
            q_g = (gauged["q_proj"] @ x).reshape(n_heads, d_head, 2, 8).permute(2, 0, 3, 1)
            k_g = (gauged["k_proj"] @ x).reshape(n_kv_heads, d_head, 2, 8).permute(2, 0, 3, 1)
            repeat = n_heads // n_kv_heads
            q_row_ratio = torch.linalg.vector_norm(gauged["q_proj"], dim=1).reshape(n_heads, d_head)
            q_row_ratio = q_row_ratio / torch.linalg.vector_norm(weights["q_proj"], dim=1).reshape(n_heads, d_head)
            k_row_ratio = torch.linalg.vector_norm(gauged["k_proj"], dim=1).reshape(n_kv_heads, d_head)
            k_row_ratio = k_row_ratio / torch.linalg.vector_norm(weights["k_proj"], dim=1).reshape(n_kv_heads, d_head)
            torch.testing.assert_close(
                q_row_ratio.reshape(n_kv_heads, repeat, d_head).mean(dim=1) * k_row_ratio,
                torch.ones_like(k_row_ratio), atol=1e-6, rtol=1e-5,
            )
            if n_kv_heads < n_heads:
                k = k.repeat_interleave(repeat, dim=1)
                k_g = k_g.repeat_interleave(repeat, dim=1)
            torch.testing.assert_close(q @ k.transpose(-2, -1), q_g @ k_g.transpose(-2, -1), atol=1e-6, rtol=1e-5)
            cos, sin = compute_llama2_rotary_emb(d_head, 8, x.device, dtype=x.dtype)
            q, k = apply_llama2_rope(q, k, cos, sin)
            q_g, k_g = apply_llama2_rope(q_g, k_g, cos, sin)
            torch.testing.assert_close(q @ k.transpose(-2, -1), q_g @ k_g.transpose(-2, -1), atol=1e-6, rtol=1e-5)
            reference, _, _ = forward_transformer_block(
                weights, x, n_heads=n_heads, n_kv_heads=n_kv_heads,
                seq_len=8, rms_norm_eps=1e-5,
            )
            actual, _, _ = forward_transformer_block(
                gauged, x, n_heads=n_heads, n_kv_heads=n_kv_heads,
                seq_len=8, rms_norm_eps=1e-5,
            )
            rel = (reference - actual).norm() / reference.norm()
            self.assertLess(rel.item(), 2e-6)
            self.assertEqual(metadata["method"], "headwise_pairwise_energy_balance_v1")

    def test_rejects_incompatible_head_counts(self):
        weights = self._weights(4, 4)
        with self.assertRaises(ValueError):
            balanced_attention_gauge(weights, 3, 2)


if __name__ == "__main__":
    unittest.main()
