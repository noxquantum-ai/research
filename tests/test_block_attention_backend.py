import unittest

import torch

from nox.block_compiler import (
    apply_llama2_rope, compute_llama2_rotary_emb, scaled_dot_product_attention_2d,
)


class BlockAttentionBackendTests(unittest.TestCase):
    def test_fused_attention_matches_explicit_causal_mha_and_gqa(self):
        generator = torch.Generator().manual_seed(97)
        for kv_heads in (1, 2):
            q = torch.randn(8, 12, generator=generator)
            k = torch.randn(kv_heads * 4, 12, generator=generator)
            v = torch.randn(kv_heads * 4, 12, generator=generator)
            q4 = q.reshape(2, 4, 2, 6).permute(2, 0, 3, 1)
            k4 = k.reshape(kv_heads, 4, 2, 6).permute(2, 0, 3, 1)
            v4 = v.reshape(kv_heads, 4, 2, 6).permute(2, 0, 3, 1)
            cos, sin = compute_llama2_rotary_emb(4, 6, "cpu")
            q4, k4 = apply_llama2_rope(q4, k4, cos, sin)
            k4 = k4.repeat_interleave(2 // kv_heads, dim=1)
            v4 = v4.repeat_interleave(2 // kv_heads, dim=1)
            score = q4 @ k4.transpose(-2, -1) / 2
            future = torch.ones(6, 6, dtype=torch.bool).triu(1)
            score = score.masked_fill(future, float("-inf"))
            expected = (score.softmax(-1) @ v4).permute(1, 3, 0, 2).reshape(8, 12)
            actual = scaled_dot_product_attention_2d(q, k, v, 2, kv_heads, 6)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
