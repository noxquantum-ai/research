import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from nox.compiler import nox_hybrid_dmrg_decompose, quantize_4bit_grouped
from benchmarks.evaluate_high_fidelity_mmlu import (
    audit_mmlu_response_finiteness,
    install_pinned_mmlu_loader,
    summarize_mmlu_harness,
)
from runpod_execution.nec_high_fidelity.gpu_qualification import compare_mmlu_smokes
from benchmarks.evaluate_high_fidelity_nec import evaluation_memory_budget, save_top1_predictions
from benchmarks.qualification_heartbeat import read_progress
from runpod_execution.nec_high_fidelity.runpod_driver import (
    compile_command,
    evaluation_batch_sizes,
    evaluation_memory_limits,
    load_config,
    preflight_qualified_batch_sizes,
    validate_matched_evaluations,
    verify_result,
    verify_checkpoint_matches_run_config,
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
    def test_qualification_heartbeat_reads_durable_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            progress = Path(temporary) / "progress.json"
            progress.write_text(json.dumps({
                "status": "RUNNING",
                "phase": "compiling_projection",
                "updated_at_utc": "2026-10-20T00:00:00+00:00",
                "ignored": "large payload",
            }))
            self.assertEqual(read_progress(progress), {
                "status": "RUNNING",
                "phase": "compiling_projection",
                "updated_at_utc": "2026-10-20T00:00:00+00:00",
            })
            progress.write_text("not-json")
            self.assertIsNone(read_progress(progress))

    def test_mmlu_batch_smoke_requires_same_prompts_predictions_and_scores(self):
        sample = {
            "doc_id": 0,
            "doc_hash": "doc",
            "prompt_hash": "prompt",
            "target_hash": "target",
            "resps": [[(-1.0, True)], [(-2.0, False)], [(-3.0, False)], [(-4.0, False)]],
            "filtered_resps": [(-1.0, True), (-2.0, False), (-3.0, False), (-4.0, False)],
            "metrics": ["acc", "acc_norm"],
            "acc": 1.0,
            "acc_norm": 1.0,
        }
        serial = json.loads(json.dumps({"samples": {"mmlu_subject": [sample]}}))
        batched = json.loads(json.dumps(serial))
        for request in batched["samples"]["mmlu_subject"][0]["resps"]:
            request[0][0] += 5e-5
        for response in batched["samples"]["mmlu_subject"][0]["filtered_resps"]:
            response[0] += 5e-5
        audit = compare_mmlu_smokes(serial, batched, ["subject"])
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["numerical_likelihood_values_compared"], 4)
        self.assertAlmostEqual(audit["maximum_absolute_likelihood_delta"], 5e-5)

        near_tie_serial = json.loads(json.dumps(serial))
        near_tie_batched = json.loads(json.dumps(serial))
        near_tie_serial_row = near_tie_serial["samples"]["mmlu_subject"][0]
        near_tie_batched_row = near_tie_batched["samples"]["mmlu_subject"][0]
        near_tie_serial_row["resps"] = [[(-1.0, True)], [(-1.00005, False)], [(-3.0, False)], [(-4.0, False)]]
        near_tie_batched_row["resps"] = [[(-1.00005, True)], [(-1.0, False)], [(-3.0, False)], [(-4.0, False)]]
        with self.assertRaisesRegex(RuntimeError, "predicted answer choice changed"):
            compare_mmlu_smokes(near_tie_serial, near_tie_batched, ["subject"])

        batched["samples"]["mmlu_subject"][0]["resps"][0][0][0] += 1e-2
        with self.assertRaisesRegex(RuntimeError, "exceeding 0.0001"):
            compare_mmlu_smokes(serial, batched, ["subject"])

    def test_top1_prediction_sidecar_is_exact_and_hash_audited(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.i32"
            metadata = save_top1_predictions(torch.tensor([1, 31999, 7]), path)
            self.assertEqual(metadata["count"], 3)
            self.assertEqual(metadata["bytes"], 12)
            self.assertEqual(metadata["sha256"], sha256_file(path))
            values = np.fromfile(path, dtype="<i4")
            self.assertEqual(values.tolist(), [1, 31999, 7])
            with self.assertRaises(FileExistsError):
                save_top1_predictions(torch.tensor([0]), path)

    def test_evaluation_memory_caps_keep_legacy_default_and_allow_runpod_override(self):
        import os

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                evaluation_memory_budget(2),
                {0: "10GiB", 1: "10GiB", "cpu": "24GiB"},
            )
        with patch.dict(os.environ, {
            "NEC_EVAL_GPU_MEMORY_GIB": "60",
            "NEC_EVAL_CPU_MEMORY_GIB": "48",
        }, clear=True):
            self.assertEqual(
                evaluation_memory_budget(1),
                {0: "60GiB", "cpu": "48GiB"},
            )
        with patch.dict(os.environ, {"NEC_EVAL_GPU_MEMORY_GIB": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                evaluation_memory_budget(1)

    def test_runpod_memory_caps_adapt_to_hardware_and_allow_explicit_override(self):
        gib = 1024**3
        self.assertEqual(
            evaluation_memory_limits([24 * gib], 32 * gib, {}),
            (18, 24),
        )
        self.assertEqual(
            evaluation_memory_limits([80 * gib], 64 * gib, {}),
            (60, 48),
        )
        self.assertEqual(
            evaluation_memory_limits(
                [24 * gib], 32 * gib,
                {"NEC_EVAL_GPU_MEMORY_GIB": "16", "NEC_EVAL_CPU_MEMORY_GIB": "20"},
            ),
            (16, 20),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluation_memory_limits([24 * gib], 32 * gib, {"NEC_EVAL_GPU_MEMORY_GIB": "0"})

    def test_runpod_batch_sizes_adapt_once_to_least_free_gpu(self):
        gib = 1024**3
        self.assertEqual(evaluation_batch_sizes([80 * gib], {}), (8, 16))
        self.assertEqual(evaluation_batch_sizes([48 * gib], {}), (4, 8))
        self.assertEqual(evaluation_batch_sizes([34 * gib], {}), (2, 4))
        self.assertEqual(evaluation_batch_sizes([24 * gib], {}), (1, 2))
        self.assertEqual(evaluation_batch_sizes([80 * gib, 40 * gib], {}), (2, 4))
        self.assertEqual(evaluation_batch_sizes(
            [80 * gib], {"NEC_PPL_BATCH_SIZE": "3", "NEC_MMLU_BATCH_SIZE": "7"},
        ), (3, 7))
        with self.assertRaisesRegex(ValueError, "positive"):
            evaluation_batch_sizes([80 * gib], {"NEC_PPL_BATCH_SIZE": "0"})

    def test_runpod_reuse_requires_same_checkpoint_and_source_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            result.write_text(json.dumps({
                "status": "COMPLETE",
                "checkpoint_manifest_sha256": "checkpoint-hash",
                "source_commit": "source-commit",
                "model_variant": "high_fidelity_nec",
            }))
            self.assertIsNotNone(verify_result(
                result, "checkpoint-hash", source_commit_hash="source-commit",
                model_variant="high_fidelity_nec",
            ))
            with self.assertRaisesRegex(RuntimeError, "different source commit"):
                verify_result(result, "checkpoint-hash", source_commit_hash="other-commit")
            with self.assertRaisesRegex(RuntimeError, "different checkpoint"):
                verify_result(result, "other-checkpoint")
            with self.assertRaisesRegex(RuntimeError, "different model variant"):
                verify_result(result, "checkpoint-hash", model_variant="dense")

    def test_evaluation_requires_batches_qualified_for_same_source_and_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {
                "status": "PASS",
                "source_commit": "commit",
                "run_config": {"sha256": "config"},
                "gpu_qualification": {
                    "status": "PASS",
                    "source_commit": "commit",
                    "run_config_sha256": "config",
                    "ppl_batch_equivalence": {"status": "PASS", "qualified_batch_size": 8},
                    "mmlu_gpu_qualification": {"status": "PASS", "qualified_batch_size": 16},
                    "compiler_capacity_smoke": {"status": "PASS"},
                    "evaluation_memory_caps_gib": {"gpu_per_device_gib": 60, "cpu_gib": 48},
                },
                "evaluation_memory_caps_gib": {"gpu_per_device": 60, "cpu": 48},
            }
            (root / "runpod_preflight.json").write_text(json.dumps(record))
            self.assertEqual(
                preflight_qualified_batch_sizes(root, "commit", "config"),
                (8, 16, 60, 48),
            )
            with self.assertRaisesRegex(RuntimeError, "does not qualify"):
                preflight_qualified_batch_sizes(root, "other", "config")

    def test_matched_comparison_requires_identical_token_and_task_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.i32"
            predictions.write_bytes(torch.arange(9, dtype=torch.int32).numpy().astype("<i4").tobytes())
            metric = {
                "input_tokens": 10,
                "scored_tokens": 9,
                "token_ids_sha256": "tokens",
                "bos_token_id": 1,
                "inference_batch_size": 2,
                "window_specification": [{"begin": 0, "end": 10}],
                "top1_prediction_sidecar": {
                    "path": str(predictions),
                    "sha256": sha256_file(predictions),
                    "dtype": "int32_little_endian",
                    "count": 9,
                    "bytes": 36,
                },
            }
            compatibility = {"model_config": {"rms_norm_eps": 1e-5}, "tokenizer": {"vocab_size": 32000}}
            dense_ppl = {
                "protocol_fingerprint": "ppl-protocol",
                "metrics": {"primary_10k": dict(metric), "full_test": dict(metric)},
                "model_tokenizer_compatibility": compatibility,
                "independent_numerical_coefficients": 100,
            }
            nec_ppl = json.loads(json.dumps(dense_ppl))
            subject = {"accuracy": 0.5, "questions": 2}
            dense_mmlu = {
                "protocol_fingerprint": "mmlu-protocol",
                "evaluation_input_identity": {"samples": 2, "sha256": "mmlu-inputs"},
                "per_subject": {"subject": dict(subject)},
                "model_tokenizer_compatibility": compatibility,
                "independent_numerical_coefficients": 100,
            }
            nec_mmlu = json.loads(json.dumps(dense_mmlu))
            audit = validate_matched_evaluations(dense_ppl, nec_ppl, dense_mmlu, nec_mmlu)
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["wikitext2_dense_top1_prediction_agreement"]["full_test"], 1.0)
            nec_ppl["metrics"]["primary_10k"]["token_ids_sha256"] = "different"
            with self.assertRaisesRegex(RuntimeError, "token_ids_sha256"):
                validate_matched_evaluations(dense_ppl, nec_ppl, dense_mmlu, nec_mmlu)

    def test_runpod_checkpoint_configuration_is_checked_against_pins(self):
        config = load_config()
        self.assertEqual(config["calibration"]["calibration_tokens"], 65_536)
        self.assertEqual(config["calibration"]["heldout_tokens"], 16_384)
        self.assertEqual(config["calibration"]["sampling"], "uniform")
        self.assertEqual(config["calibration"]["compiler_backend"], "full_matrix")
        command = compile_command(config, Path("/tmp/nec-checkpoint"), resume=False)
        self.assertEqual(
            command[command.index("--calibration-backend") + 1], "full_matrix",
        )
        self.assertEqual(
            command[command.index("--calibration-windows-per-chunk") + 1], "4",
        )
        self.assertEqual(
            command[command.index("--statistics-dtype") + 1], "float64",
        )
        self.assertEqual(
            config["acceptance"]["primary_10k_next_token_accuracy_retention_min"], 0.90,
        )
        qualification_script = (
            Path(__file__).resolve().parents[1]
            / "runpod_execution" / "nec_high_fidelity" / "qualify_streaming.sh"
        ).read_text()
        self.assertNotIn("timeout --foreground", qualification_script)
        self.assertIn('HEARTBEAT_SECONDS="${NEC_QUAL_HEARTBEAT_SECONDS:-60}"', qualification_script)
        self.assertIn("qualification_heartbeat.py", qualification_script)
        self.assertIn("require_down_gate", qualification_script)
        self.assertNotIn("run_variant reference-full-matrix", qualification_script)
        self.assertIn("down-reference)", qualification_script)
        self.assertIn("down-fp32)", qualification_script)
        self.assertIn("NEC_BOOTSTRAP_PROFILE=qualification", qualification_script)
        bootstrap_script = (
            Path(__file__).resolve().parents[1]
            / "runpod_execution" / "nec_high_fidelity" / "bootstrap.sh"
        ).read_text()
        self.assertIn('BOOTSTRAP_PROFILE="${NEC_BOOTSTRAP_PROFILE:-production}"', bootstrap_script)
        self.assertIn('"uv==0.13.1"', bootstrap_script)
        self.assertIn("uv python install 3.13.15", bootstrap_script)
        compression = config["compression"]
        identity = {
            "config": {
                "ranks": compression["ranks"],
                "r_bridge": compression["bridge_rank"],
                "n_outliers": compression["sparse_outlier_budget"],
                "mpo_n_sweeps": compression["mpo_sweeps"],
                "mpo_alpha_awt": compression["activation_scale_exponent"],
                "diagonal_prior_strength": compression["diagonal_prior_strength"],
                "representation": compression["representation"],
                "epsilon": 0.005,
            },
            "attention_gauge": compression["attention_gauge"],
            "mlp_gauge": compression["mlp_gauge"],
            "mlp_objective": compression["mlp_objective"],
            "budget_allocation": compression["budget_allocation"],
            "defect_bridge_mode": compression["defect_bridge"],
            "pees_rank": compression["pees_rank"],
            "mpo_site_order": compression["site_order"],
            "calibration_activation_mode": "dense",
            "construction_mode": compression["construction_mode"],
            "input_data": {
                "calibration_token_count": config["calibration"]["calibration_tokens"],
                "heldout_token_count": config["calibration"]["heldout_tokens"],
                "activation_sequence_length": config["calibration"]["sequence_length"],
                "calibration_sampling": config["calibration"]["sampling"],
                "calibration_seed": config["calibration"]["calibration_seed"],
                "validation_sampling": config["calibration"]["sampling"],
                "validation_seed": config["calibration"]["heldout_seed"],
                "tokenizer_model_id": config["model"]["id"],
                "tokenizer_model_revision": config["model"]["revision"],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            manifest_path = checkpoint / "model_manifest.json"
            manifest_path.write_text(json.dumps({"run_identity": identity}))
            verify_checkpoint_matches_run_config(checkpoint, config)
            identity["input_data"]["calibration_seed"] = -1
            manifest_path.write_text(json.dumps({"run_identity": identity}))
            with self.assertRaisesRegex(RuntimeError, "calibration_seed"):
                verify_checkpoint_matches_run_config(checkpoint, config)

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

    def test_mmlu_response_audit_rejects_nonfinite_likelihoods(self):
        results = {"samples": {"mmlu_a": [{
            "resps": [[(-1.0, True)]],
            "filtered_resps": [(-1.0, True)],
        }]}}
        self.assertEqual(
            audit_mmlu_response_finiteness(results, ["a"])["status"], "PASS",
        )
        results["samples"]["mmlu_a"][0]["resps"] = [[(float("nan"), False)]]
        with self.assertRaises(FloatingPointError):
            audit_mmlu_response_finiteness(results, ["a"])

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
                "calibration_token_count": 65_536,
                "heldout_token_count": 16_384,
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
                with self.assertRaisesRegex(ValueError, "requested protocol"):
                    validate_high_fidelity_checkpoint(directory)
                report = validate_high_fidelity_checkpoint(
                    directory,
                    expected_calibration_tokens=65_536,
                    expected_heldout_tokens=16_384,
                )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["projection_coefficients"], 32 * 7 * 17_219)
            self.assertEqual(
                report["independent_numerical_coefficients"],
                32 * 7 * 17_219 + 32 * 8192 + UNCHANGED_COEFFICIENTS,
            )


if __name__ == "__main__":
    unittest.main()
