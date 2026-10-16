import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nox.sequential_compiler import forward_transformer_block


class DenseLayerParityTests(unittest.TestCase):
    def test_small_embedding_inputs_match_hf_with_model_epsilon(self):
        torch.manual_seed(41)
        config = LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            rms_norm_eps=1e-5, attention_dropout=0.0,
        )
        model = LlamaForCausalLM(config).float().eval()
        with torch.no_grad():
            model.model.embed_tokens.weight.mul_(0.01)
            states = model(torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), output_hidden_states=True).hidden_states
            layer = model.model.layers[0]
            weights = {name: getattr(layer.self_attn, name).weight for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
            weights.update({name: getattr(layer.mlp, name).weight for name in ("gate_proj", "up_proj", "down_proj")})
            weights.update({name: getattr(layer, name).weight for name in ("input_layernorm", "post_attention_layernorm")})
            x = states[0].squeeze(0).T
            reference = states[1].squeeze(0).T
            actual, _, _ = forward_transformer_block(weights, x, n_heads=2, n_kv_heads=1, seq_len=7, rms_norm_eps=config.rms_norm_eps)
            legacy, _, _ = forward_transformer_block(weights, x, n_heads=2, n_kv_heads=1, seq_len=7, rms_norm_eps=1e-6)
        torch.testing.assert_close(actual, reference, atol=1e-7, rtol=1e-5)
        self.assertGreater(((legacy - reference).norm() / reference.norm()).item(), 0.05)


if __name__ == "__main__":
    unittest.main()
