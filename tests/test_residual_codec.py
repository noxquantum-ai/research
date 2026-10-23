import tempfile
import types
import unittest
from pathlib import Path

import torch

from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints
from nox.compiler import reconstruct_from_artifact
from nox.packing import pack_artifact
from nox.pees import layer_projection_trajectories
from nox.residual_codec import (
    CODEC_TYPE,
    codec_payload_bytes,
    decode_groupwise,
    encode_groupwise,
)
from nox.sequential_compiler import forward_transformer_block
from scripts.compile_direct_packed import projection_contexts_and_output
from scripts.run_residual_codec_pilot import block_error_metrics, load_activation_tensor


class PackedResidualCodecTests(unittest.TestCase):
    def test_signed_levels_round_trip_for_all_supported_widths(self):
        for bits in (2, 3, 4, 5, 6, 8):
            qmax = (1 << (bits - 1)) - 1
            qmin = -qmax
            values = torch.tensor(
                [[qmin, qmin + 1, 0, qmax, qmax, -1, 1]], dtype=torch.float32,
            )
            artifact = encode_groupwise(values, bits=bits, group_size=4)
            decoded = decode_groupwise(artifact)
            self.assertTrue(torch.equal(values, decoded))
            self.assertEqual(artifact["qmin"], qmin)
            self.assertEqual(artifact["qmax"], qmax)

    def test_nonmultiple_columns_are_grouped_row_by_row(self):
        matrix = torch.arange(3 * 70, dtype=torch.float32).reshape(3, 70) - 50
        artifact = encode_groupwise(matrix, bits=3, group_size=64)
        self.assertEqual(artifact["groups_per_row"], 2)
        self.assertEqual(artifact["packed_codes"].numel(), 3 * 2 * 64 * 3 // 8)
        self.assertEqual(tuple(decode_groupwise(artifact).shape), (3, 70))

    def test_zero_groups_decode_as_zero(self):
        matrix = torch.zeros(2, 67)
        artifact = encode_groupwise(matrix, bits=2, group_size=64)
        self.assertTrue(torch.equal(decode_groupwise(artifact), matrix))
        self.assertTrue(torch.equal(artifact["group_scales"], torch.zeros_like(artifact["group_scales"])))

    def test_activation_scales_use_only_given_calibration_statistics(self):
        matrix = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        rms = torch.tensor([0.25, 1.0, 4.0, 1000.0])
        artifact = encode_groupwise(
            matrix, bits=4, group_size=4, activation_rms=rms,
            alpha=0.5, min_scale=0.5, max_scale=2.0,
        )
        self.assertTrue(torch.equal(artifact["column_scales"], torch.tensor([0.5, 1.0, 2.0, 2.0], dtype=torch.float16)))
        self.assertTrue(torch.isfinite(decode_groupwise(artifact)).all())

    def test_payload_byte_count_includes_codes_and_all_scales_exactly(self):
        matrix = torch.randn(5, 70)
        rms = torch.rand(70) + 0.1
        artifact = encode_groupwise(
            matrix, bits=3, group_size=64, activation_rms=rms, alpha=0.5,
        )
        expected = (
            artifact["packed_codes"].numel() * artifact["packed_codes"].element_size()
            + artifact["group_scales"].numel() * 2
            + artifact["column_scales"].numel() * 2
        )
        self.assertEqual(codec_payload_bytes(artifact), expected)
        self.assertEqual(artifact["payload_bytes"], expected)

    def test_fp16_reconstruction_is_cast_of_fp32_reconstruction(self):
        artifact = encode_groupwise(torch.randn(4, 73), bits=4, group_size=64)
        fp32 = decode_groupwise(artifact, dtype=torch.float32)
        fp16 = decode_groupwise(artifact, dtype=torch.float16)
        self.assertTrue(torch.equal(fp16, fp32.to(torch.float16)))

    def test_nonfinite_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            encode_groupwise(torch.tensor([[float("nan")]]), bits=2)

    def test_saved_dense_validation_payload_loads_its_x_test_matrix(self):
        expected = torch.randn(8, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense_input_layer_0.pt"
            torch.save({"X_test": expected, "rms_norm_eps": 1e-5}, path)
            actual = load_activation_tensor(path)
        self.assertTrue(torch.equal(actual, expected))

    def test_training_statistics_can_collect_q_k_and_up_contexts(self):
        weights = {
            "q_proj": torch.randn(8, 8),
            "k_proj": torch.randn(8, 8),
            "v_proj": torch.randn(8, 8),
            "o_proj": torch.randn(8, 8),
            "gate_proj": torch.randn(12, 8),
            "up_proj": torch.randn(12, 8),
            "down_proj": torch.randn(8, 12),
            "input_layernorm": torch.ones(8),
            "post_attention_layernorm": torch.ones(8),
        }
        inputs = torch.randn(8, 8)
        contexts = layer_projection_trajectories(
            weights, weights, inputs, inputs,
            seq_len=4, n_heads=2, n_kv_heads=2, rms_norm_eps=1e-5,
            max_tokens=8, projections=("q_proj", "k_proj", "up_proj"),
        )
        self.assertEqual(set(contexts), {"q_proj", "k_proj", "up_proj"})
        self.assertEqual(tuple(contexts["q_proj"][0].shape), (8, 8))
        self.assertEqual(tuple(contexts["up_proj"][0].shape), (8, 8))

    def test_block_diagnostics_handle_a_four_layer_prefix(self):
        torch.manual_seed(71)
        dense_weights = {}
        candidate_weights = {}
        validation_inputs = {}
        for layer_index in range(4):
            weights = {
                "q_proj": torch.randn(8, 8),
                "k_proj": torch.randn(8, 8),
                "v_proj": torch.randn(8, 8),
                "o_proj": torch.randn(8, 8),
                "gate_proj": torch.randn(12, 8),
                "up_proj": torch.randn(12, 8),
                "down_proj": torch.randn(8, 12),
                "input_layernorm": torch.ones(8),
                "post_attention_layernorm": torch.ones(8),
            }
            dense_weights[layer_index] = weights
            candidate_weights[layer_index] = {
                name: tensor.clone()
                for name, tensor in weights.items()
                if name in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
            }
            validation_inputs[layer_index] = torch.randn(8, 8)
        metrics = block_error_metrics(
            candidate_weights,
            dense_weights,
            validation_inputs,
            layer_indices=[0, 1, 2, 3],
            seq_len=4,
            heads=2,
            kv_heads=2,
            rms_norm_eps=1e-5,
        )
        self.assertIn("layers_0_3_composed", metrics)
        self.assertEqual(metrics["layers_0_3_composed"]["relative_error"], 0.0)

    def test_full_constructor_contexts_match_dense_block(self):
        torch.manual_seed(19)
        width, intermediate, tokens, seq_len = 8, 12, 4, 2
        weights = {
            "q_proj": torch.randn(width, width),
            "k_proj": torch.randn(width, width),
            "v_proj": torch.randn(width, width),
            "o_proj": torch.randn(width, width),
            "gate_proj": torch.randn(intermediate, width),
            "up_proj": torch.randn(intermediate, width),
            "down_proj": torch.randn(width, intermediate),
            "input_layernorm": torch.randn(width),
            "post_attention_layernorm": torch.randn(width),
        }
        hidden = torch.randn(width, tokens)
        contexts, actual = projection_contexts_and_output(
            weights, hidden, seq_len=seq_len, n_heads=2, n_kv_heads=2,
            rms_norm_eps=1e-5,
        )
        expected, _, _ = forward_transformer_block(
            weights, hidden, seq_len=seq_len, n_heads=2, n_kv_heads=2,
            rms_norm_eps=1e-5,
        )
        self.assertEqual(set(contexts), {
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        })
        self.assertEqual(contexts["down_proj"].shape, (intermediate, tokens))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_direct_dispatch_and_hybrid_dispatch(self):
        matrix = torch.randn(8, 8)
        packed = encode_groupwise(matrix, bits=4, group_size=4)
        self.assertEqual(packed["codec_type"], CODEC_TYPE)
        self.assertTrue(torch.equal(reconstruct_from_artifact(packed), decode_groupwise(packed)))
        base = encode_groupwise(torch.randn(8, 8), bits=4, group_size=4)
        residual = encode_groupwise(torch.randn(8, 8), bits=3, group_size=4)
        hybrid = {
            "codec_type": "nec_hybrid_residual_v1",
            "base_artifact": base,
            "residual_codec": residual,
            "orig_shape": (8, 8),
        }
        actual = reconstruct_from_artifact(hybrid)
        expected = decode_groupwise(base) + decode_groupwise(residual)
        self.assertTrue(torch.equal(actual, expected))

    def test_packed_nec_base_dispatches_inside_hybrid(self):
        base = {
            "cores": [torch.arange(-8, 8, dtype=torch.float16).reshape(1, 4, 4, 1)],
            "d_awt": torch.ones(4, dtype=torch.float16),
            "A_bridge": None,
            "B_bridge": None,
            "orig_shape": (4, 4),
            "quantization_bits": 4,
            "sparse_indices": None,
            "sparse_values": None,
        }
        packed_base = pack_artifact(base)
        residual = encode_groupwise(torch.randn(4, 4), bits=2, group_size=4)
        hybrid = {
            "codec_type": "nec_hybrid_residual_v1",
            "base_artifact": packed_base,
            "residual_codec": residual,
        }
        actual = reconstruct_from_artifact(hybrid)
        expected = reconstruct_from_artifact(packed_base) + decode_groupwise(residual)
        self.assertTrue(torch.equal(actual, expected))

    def test_codec_checkpoint_injects_into_tiny_model(self):
        class TinyLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = torch.nn.Module()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(self.self_attn, name, torch.nn.Linear(8, 8, bias=False))
                self.mlp = torch.nn.Module()
                self.mlp.gate_proj = torch.nn.Linear(8, 12, bias=False)
                self.mlp.up_proj = torch.nn.Linear(8, 12, bias=False)
                self.mlp.down_proj = torch.nn.Linear(12, 8, bias=False)
                self.input_layernorm = torch.nn.LayerNorm(8)
                self.post_attention_layernorm = torch.nn.LayerNorm(8)

        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.layers = torch.nn.ModuleList([TinyLayer()])
        targets = {
            "q_proj": (model.model.layers[0].self_attn.q_proj, (8, 8)),
            "k_proj": (model.model.layers[0].self_attn.k_proj, (8, 8)),
            "v_proj": (model.model.layers[0].self_attn.v_proj, (8, 8)),
            "o_proj": (model.model.layers[0].self_attn.o_proj, (8, 8)),
            "gate_proj": (model.model.layers[0].mlp.gate_proj, (12, 8)),
            "up_proj": (model.model.layers[0].mlp.up_proj, (12, 8)),
            "down_proj": (model.model.layers[0].mlp.down_proj, (8, 12)),
        }
        artifacts = {}
        expected = {}
        for name, (_, shape) in targets.items():
            weight = torch.randn(shape)
            codec = encode_groupwise(weight, bits=3, group_size=4)
            artifacts[name] = codec
            expected[name] = decode_groupwise(codec)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compressed_layer_0.pt"
            torch.save({"layer_idx": 0, "compressed_artifacts": artifacts}, path)
            injected = inject_nox_checkpoints(model, directory, device="cpu")
        self.assertEqual(injected, [0])
        for name, (module, _) in targets.items():
            self.assertTrue(torch.equal(module.weight, expected[name]), name)


if __name__ == "__main__":
    unittest.main()
