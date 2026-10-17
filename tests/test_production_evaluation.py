import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch

from nox.compiler import nox_hybrid_dmrg_decompose, quantize_4bit_grouped
from benchmarks.evaluate_high_fidelity_mmlu import (
    install_pinned_mmlu_loader,
    summarize_mmlu_harness,
)
from nox.artifacts import fingerprint
from nox.evaluation import (
    MMLU_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    UNCHANGED_COEFFICIENTS,
    WIKITEXT_REVISION,
    sha256_file,
    validate_high_fidelity_checkpoint,
    validate_llama2_model_tokenizer,
    validate_mpo_artifact,
)


def tiny_artifact():
    precision = {
        "sparse_shield": 4,
        "bridge_A": 2 * 128,
        "bridge_B": 128 * 2,
        "mpo_site_0": 64,
        "mpo_site_1": 64 * 1 * 1 * 128,
        "mpo_site_2": 128 * 1 * 1 * 64,
        "mpo_site_3": 64 * 2 * 2,
    }
    precision = {
        name: {
            "group_size": 64,
            "input_coefficients": count,
            "q4_clipped_code_count": 0,
            "q4_nonzero_input_rounded_to_zero_count": 0,
            "q4_positive_max_code_count": 0,
            "q4_negative_min_code_count": 0,
            "fp16_nonzero_value_underflow_to_zero_count": 0,
            "fp16_subnormal_count": 0,
            "fp16_overflow_count": 0,
            "fp16_nonfinite_count": 0,
            "input_max_abs": 0.0,
            "dequantized_max_abs": 0.0,
            "stored_fp16_max_abs": 0.0,
        }
        for name, count in precision.items()
    }
    return {
        "codec_type": "nec_mpo_v1",
        "orig_shape": (2, 2),
        "cores": [
            torch.ones(1, 1, 1, 64, dtype=torch.float16),
            torch.ones(64, 1, 1, 128, dtype=torch.float16),
            torch.ones(128, 1, 1, 64, dtype=torch.float16),
            torch.ones(64, 2, 2, 1, dtype=torch.float16),
        ],
        "A_bridge": torch.ones(2, 128, dtype=torch.float16),
        "B_bridge": torch.ones(128, 2, dtype=torch.float16),
        "d_awt": torch.ones(2, dtype=torch.float16),
        "sparse_indices": torch.tensor([[0, 1]], dtype=torch.int32),
        "sparse_values": torch.tensor([0.25], dtype=torch.float16),
        "compiler_hyperparameters": {
            "n_sweeps": 2,
            "prior_gamma": 0.01,
            "alpha_awt": 0.5,
            "chunk_size": 32,
            "diagonal_prior_strength": 0.0,
            "requested_bridge_rank": 128,
            "requested_sparse_budget": 524288,
            "realized_sparse_count": 1,
            "quantize_components": True,
        },
        "quantization_bits": 4,
        "core_quantization_bits": 4,
        "precision_diagnostics": precision,
    }


class ProductionEvaluationTests(unittest.TestCase):
    def test_harness_dataset_loader_forces_pinned_mmlu_revision(self):
        class FakeDatasets:
            def __init__(self):
                self.received = []

            def load_dataset(self, path, *args, **kwargs):
                self.received.append((path, kwargs))
                return "fixture"

        module = FakeDatasets()
        calls, original = install_pinned_mmlu_loader(module)
        self.assertEqual(module.load_dataset("cais/mmlu", name="abstract_algebra"), "fixture")
        self.assertEqual(module.received[0][1]["revision"], MMLU_REVISION)
        self.assertEqual(calls[0]["revision"], MMLU_REVISION)
        module.load_dataset = original

    def test_harness_summary_is_question_weighted_and_complete(self):
        subjects = ["a", "b"]
        results = {
            "results": {
                "mmlu_a": {"acc,none": 0.5},
                "mmlu_b": {"acc,none": 1.0},
            },
            "n-samples": {
                "mmlu_a": {"effective": 2},
                "mmlu_b": {"effective": 1},
            },
            "groups": {"mmlu": {"acc,none": 2.0 / 3.0}},
        }
        summary = summarize_mmlu_harness(results, subjects, full_test=False)
        self.assertEqual(summary["total_questions"], 3)
        self.assertAlmostEqual(summary["overall_accuracy"], 2.0 / 3.0)

    def test_mpo_artifact_count_is_derived_from_tensors(self):
        report = validate_mpo_artifact("fixture", tiny_artifact(), (2, 2))
        self.assertEqual(report["independent_numerical_coefficients"], 17_219)
        self.assertEqual(report["bridge_rank"], 128)
        self.assertEqual(report["sparse_coefficients"], 1)

    def test_q4_and_fp16_precision_diagnostics_separate_clipping_and_underflow(self):
        quantized, diagnostics = quantize_4bit_grouped(
            torch.tensor([1e-8, 0.0, 0.0, 0.0]),
            group_size=4,
            return_diagnostics=True,
        )
        self.assertTrue(torch.isfinite(quantized).all())
        self.assertEqual(diagnostics["q4_clipped_code_count"], 0)
        self.assertEqual(diagnostics["q4_positive_max_code_count"], 1)
        self.assertEqual(diagnostics["q4_negative_min_code_count"], 0)
        self.assertEqual(diagnostics["q4_nonzero_input_rounded_to_zero_count"], 0)
        self.assertEqual(diagnostics["fp16_nonzero_value_underflow_to_zero_count"], 1)
        self.assertEqual(diagnostics["fp16_overflow_count"], 0)
        self.assertEqual(diagnostics["fp16_nonfinite_count"], 0)

    def test_compiler_serializes_quantization_audit_for_every_component(self):
        torch.manual_seed(19)
        weight = torch.randn(16, 16)
        inputs = torch.randn(16, 16)
        _, _, _, _, artifact = nox_hybrid_dmrg_decompose(
            W=weight,
            X_calib=inputs,
            Y_dense=weight @ inputs,
            o_modes=(2, 2, 2, 2),
            i_modes=(2, 2, 2, 2),
            ranks=(1, 2, 4, 2, 1),
            r_bridge=2,
            n_outliers=4,
            n_sweeps=1,
            alpha_awt=0.5,
            return_artifacts=True,
            quantize_components=True,
            verbose=False,
        )
        expected = {
            "sparse_shield", "bridge_A", "bridge_B",
            "mpo_site_0", "mpo_site_1", "mpo_site_2", "mpo_site_3",
        }
        self.assertEqual(set(artifact["precision_diagnostics"]), expected)
        self.assertTrue(all(
            item["q4_clipped_code_count"] == 0
            and item["fp16_overflow_count"] == 0
            and item["fp16_nonfinite_count"] == 0
            for item in artifact["precision_diagnostics"].values()
        ))

    def test_llama2_runtime_guard_checks_architecture_and_tokenizer(self):
        class Tokenizer:
            vocab_size = 32000
            bos_token_id = 1
            eos_token_id = 2
            unk_token_id = 0
            pad_token_id = 0
            is_fast = True
            add_bos_token = True
            add_eos_token = False
            name_or_path = "NousResearch/Llama-2-7b-hf"
            special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}

            def __len__(self):
                return 32000

            def __call__(self, text, add_special_tokens=True):
                return {
                    "input_ids": [1, 405, 11206, 5993, 3950, 24521, 410, 915, 29889]
                    if add_special_tokens else [405, 11206, 5993, 3950, 24521, 410, 915, 29889]
                }

        class Module:
            def __init__(self, shape):
                self.weight = SimpleNamespace(shape=shape)

        config = SimpleNamespace(
            model_type="llama", hidden_size=4096, intermediate_size=11008,
            num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32,
            vocab_size=32000, max_position_embeddings=4096, rms_norm_eps=1e-5,
            tie_word_embeddings=False, bos_token_id=1, eos_token_id=2,
            pad_token_id=0,
            architectures=["LlamaForCausalLM"], _commit_hash=MODEL_REVISION,
        )
        layers = []
        for _ in range(32):
            layer = SimpleNamespace(
                self_attn=SimpleNamespace(**{
                    name: Module((4096, 4096))
                    for name in ("q_proj", "k_proj", "v_proj", "o_proj")
                }),
                mlp=SimpleNamespace(**{
                    name: Module(shape)
                    for name, shape in (
                        ("gate_proj", (11008, 4096)),
                        ("up_proj", (11008, 4096)),
                        ("down_proj", (4096, 11008)),
                    )
                }),
            )
            layers.append(layer)
        model = SimpleNamespace(
            config=config,
            model=SimpleNamespace(layers=layers),
            get_input_embeddings=lambda: Module((32000, 4096)),
            get_output_embeddings=lambda: Module((32000, 4096)),
        )
        report = validate_llama2_model_tokenizer(model, Tokenizer())
        self.assertEqual(report["projection_layers"], 32)
        self.assertEqual(
            report["tokenizer"]["special_token_ids"],
            {"bos": 1, "eos": 2, "unk": 0, "pad": 0},
        )
        bad_tokenizer = Tokenizer()
        bad_tokenizer.vocab_size = 31999
        with self.assertRaisesRegex(ValueError, "vocabulary mismatch"):
            validate_llama2_model_tokenizer(model, bad_tokenizer)

    def test_mpo_artifact_rejects_duplicate_sparse_coordinates(self):
        artifact = tiny_artifact()
        artifact["sparse_indices"] = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        artifact["sparse_values"] = torch.ones(2, dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "duplicate sparse"):
            validate_mpo_artifact("fixture", artifact, (2, 2))

    def test_full_checkpoint_audit_requires_consistent_32_layer_bundle(self):
        projection_shapes = {name: (2, 2) for name in (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        )}
        identity = {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "compression_path": "no_healing_independent",
            "calibration_activation_mode": "dense",
            "construction_mode": "independent",
            "config": {
                "ranks": [1, 64, 128, 64, 1],
                "r_bridge": 128,
                "n_outliers": 524288,
            },
            "mpo_n_sweeps": 2,
            "mpo_alpha_awt": 0.5,
            "forward_options": {
                "n_heads": 32,
                "n_kv_heads": 32,
                "seq_len": 128,
                "rms_norm_eps": 1e-5,
            },
            "input_data": {
                "calibration_token_count": 256,
                "heldout_token_count": 128,
                "calibration_source": {"dataset_revision": WIKITEXT_REVISION},
                "heldout_source": {"dataset_revision": WIKITEXT_REVISION},
            },
            "layers": list(range(32)),
        }
        run_fingerprint = fingerprint(identity)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            sizes = {}
            hashes = {}
            intermediate = {}
            for layer in range(32):
                path = directory / f"compressed_layer_{layer}.pt"
                torch.save({
                    "layer_idx": layer,
                    "run_fingerprint": run_fingerprint,
                    "compressed_artifacts": {
                        name: tiny_artifact() for name in projection_shapes
                    },
                    "input_layernorm": torch.ones(4096),
                    "post_attention_layernorm": torch.ones(4096),
                }, path)
                sizes[path.name] = path.stat().st_size
                hashes[path.name] = sha256_file(path)
                for prefix in (
                    "dense_input", "calibration_input", "layer_input_validation",
                ):
                    intermediate_path = directory / f"{prefix}_layer_{layer}.pt"
                    torch.save(torch.ones(1), intermediate_path)
                    intermediate[intermediate_path.name] = {
                        "bytes": intermediate_path.stat().st_size,
                        "sha256": sha256_file(intermediate_path),
                    }
            manifest = {
                "run_identity": identity,
                "run_fingerprint": run_fingerprint,
                "checkpoint_file_sizes": sizes,
                "checkpoint_file_sha256": hashes,
                "serialized_checkpoint_bytes": sum(sizes.values()),
                "intermediate_files": intermediate,
            }
            (directory / "model_manifest.json").write_text(json.dumps(manifest))
            with patch("nox.evaluation.PROJECTION_SHAPES", projection_shapes):
                report = validate_high_fidelity_checkpoint(directory)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["projection_coefficients"], 32 * 7 * 17_219)
            self.assertEqual(
                report["independent_numerical_coefficients"],
                32 * 7 * 17_219 + 32 * 8192 + UNCHANGED_COEFFICIENTS,
            )


if __name__ == "__main__":
    unittest.main()
