import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from benchmarks.benchmark_day3_cascade import load_cached_tensor
from benchmarks.benchmark_day4_perplexity import (
    compute_perplexity,
    sliding_window_specifications,
)
from nox.artifacts import checkpoint_file_sizes, checkpoint_storage_bytes, fingerprint
from nox.compiler import estimate_artifact_tensor_bytes
from nox.sequential_compiler import compute_dense_argmax_metrics
from scripts.audit_checkpoint_reconstruction import compare_weights
from scripts.audit_function_aware_parameters import projection_budget, tier_counts


class PipelineAccountingTests(unittest.TestCase):
    def test_legacy_nec_counts_follow_actual_projection_overrides(self):
        high = tier_counts("high_fidelity_10x")
        production = tier_counts("production_18x")
        extreme = tier_counts("extreme_77x")
        self.assertEqual(high["whole_model_coefficients"], 915_935_232)
        self.assertEqual(production["whole_model_coefficients"], 758_907_648)
        self.assertEqual(extreme["whole_model_coefficients"], 633_763_072)
        self.assertEqual(
            projection_budget(
                "o_proj", {"r_bridge": 128, "n_outliers": 524_288}, 256,
            ),
            (153, 262_144),
        )
        self.assertEqual(
            projection_budget(
                "down_proj", {"r_bridge": 96, "n_outliers": 65_536}, 256,
            ),
            (144, 32_768),
        )

    def test_weight_audit_promotes_fp16_before_reductions(self):
        reference = torch.ones(100_000, dtype=torch.float16)
        identical = reference.clone()

        scores = compare_weights(reference, identical)

        self.assertEqual(scores["relative_frobenius_error"], 0.0)
        self.assertEqual(scores["cosine_similarity"], 1.0)
        self.assertTrue(torch.isfinite(torch.tensor(list(scores.values()))).all())

    def test_checkpoint_size_uses_serialized_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "compressed_layer_0.pt").write_bytes(b"abc")
            (root / "compressed_layer_1.pt").write_bytes(b"12345")
            self.assertEqual(
                checkpoint_file_sizes(root, [0, 1]),
                {"compressed_layer_0.pt": 3, "compressed_layer_1.pt": 5},
            )
            self.assertEqual(checkpoint_storage_bytes(root, [0, 1]), 8)

    def test_compiler_payload_estimate_matches_saved_tensor_dtypes(self):
        cores = [torch.zeros((2, 3), dtype=torch.float32)]
        bridge_a = torch.zeros((4, 2), dtype=torch.float32)
        bridge_b = torch.zeros((2, 3), dtype=torch.float32)
        d_awt = torch.zeros(5, dtype=torch.float32)
        sparse_indices = torch.zeros((2, 2), dtype=torch.int32)
        sparse_values = torch.zeros(2, dtype=torch.float16)

        estimated_bytes = estimate_artifact_tensor_bytes(
            cores, bridge_a, bridge_b, d_awt, sparse_count=2
        )
        stored_tensor_bytes = (
            sum(t.numel() * 2 for t in cores)
            + bridge_a.numel() * 2
            + bridge_b.numel() * 2
            + d_awt.numel() * 2
            + sparse_indices.numel() * sparse_indices.element_size()
            + sparse_values.numel() * sparse_values.element_size()
        )
        self.assertEqual(estimated_bytes, stored_tensor_bytes)

    def test_empty_tensor_cache_is_a_miss_and_remains_untouched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "layer_15_down_proj.pt"
            cache_path.write_bytes(b"")
            with patch("benchmarks.benchmark_day3_cascade.log_step"):
                self.assertIsNone(load_cached_tensor(str(cache_path)))
            self.assertEqual(cache_path.stat().st_size, 0)

    def test_run_fingerprint_is_stable_and_sensitive_to_input_data(self):
        first = {"tier": "production_18x", "tokens": 256, "split": "train"}
        self.assertEqual(fingerprint(first), fingerprint(dict(reversed(list(first.items())))))
        self.assertNotEqual(fingerprint(first), fingerprint({**first, "split": "valid"}))

    def test_dense_argmax_diagnostics_are_not_named_perplexity(self):
        dense = torch.tensor([[4.0, 0.0, -1.0, -2.0], [0.0, 3.0, 1.0, -1.0]])
        compressed = dense.clone()
        metrics = compute_dense_argmax_metrics(dense, compressed)
        self.assertEqual(metrics["top1_agreement_pct"], 100.0)
        self.assertEqual(metrics["top5_agreement_pct"], 100.0)
        self.assertEqual(metrics["dense_argmax_cross_entropy_dense_logits"], metrics["dense_argmax_cross_entropy_compressed_logits"])
        self.assertIn("not language-model perplexity", metrics["metric_semantics"])
        self.assertFalse(any("ppl" in key.lower() for key in metrics))

    def test_perplexity_counts_only_scored_next_token_labels(self):
        class Tokenizer:
            def __len__(self):
                return 4

            def __call__(self, text, return_tensors, add_special_tokens=True):
                return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 0, 1]]))

        class UniformModel:
            def eval(self):
                return self

            def __call__(self, input_ids):
                logits = torch.zeros((*input_ids.shape, 4), dtype=torch.float32)
                return SimpleNamespace(logits=logits)

        with patch("benchmarks.benchmark_day4_perplexity.log_step"):
            ppl, scored_tokens, details = compute_perplexity(
                UniformModel(),
                Tokenizer(),
                "ignored",
                seq_len=4,
                stride=3,
                max_eval_tokens=5,
                device="cpu",
                return_details=True,
            )
        self.assertEqual(scored_tokens, 4)
        self.assertAlmostEqual(ppl, 4.0, places=5)
        self.assertEqual(details["top1_predictions"].tolist(), [0, 0, 0, 0])
        self.assertEqual(details["next_token_accuracy"], 0.25)

    def test_sliding_windows_start_full_and_score_every_target_once(self):
        windows, limit = sliding_window_specifications(
            10_317, seq_len=2048, stride=512, max_eval_tokens=10_000,
        )
        self.assertEqual(limit, 10_000)
        self.assertEqual(windows[0], {
            "begin": 0, "end": 2048, "target_length": 2048,
            "scored_tokens": 2047,
        })
        self.assertEqual(windows[1], {
            "begin": 512, "end": 2560, "target_length": 512,
            "scored_tokens": 512,
        })
        self.assertEqual(windows[-1]["end"], 10_000)
        self.assertEqual(sum(window["scored_tokens"] for window in windows), 9_999)

    def test_perplexity_is_token_weighted_across_unequal_windows(self):
        class Tokenizer:
            def __len__(self):
                return 2

            def __call__(self, text, return_tensors, add_special_tokens=True):
                return SimpleNamespace(input_ids=torch.tensor([[0, 1, 0, 1, 0, 1]]))

        class WindowLossModel:
            def __init__(self):
                self.window_lengths = []

            def eval(self):
                return self

            def __call__(self, input_ids):
                self.window_lengths.append(input_ids.shape[1])
                # The correct next class gets logit 0 and the other class is
                # set to produce exactly NLL 1 or 3 for each window.
                desired_nll = 1.0 if len(self.window_lengths) == 1 else 3.0
                incorrect_logit = math.log(math.expm1(desired_nll))
                logits = torch.empty((*input_ids.shape, 2), dtype=torch.float32)
                next_class = 1 - input_ids
                logits.scatter_(-1, next_class.unsqueeze(-1), 0.0)
                logits.scatter_(-1, (1 - next_class).unsqueeze(-1), incorrect_logit)
                return SimpleNamespace(logits=logits)

        model = WindowLossModel()
        with patch("benchmarks.benchmark_day4_perplexity.log_step"):
            ppl, scored, details = compute_perplexity(
                model, Tokenizer(), "ignored", seq_len=4, stride=2,
                max_eval_tokens=6, device="cpu", return_details=True,
            )
        self.assertEqual(model.window_lengths, [4, 4])
        self.assertEqual(scored, 5)
        self.assertAlmostEqual(details["mean_nll"], 1.8)
        self.assertAlmostEqual(ppl, math.exp(1.8))

    def test_logits_reference_matches_causal_shift_and_overlapping_windows(self):
        class Tokenizer:
            def __len__(self):
                return 4

            def __call__(self, text, return_tensors, add_special_tokens=True):
                return SimpleNamespace(input_ids=torch.tensor([[0, 1, 2, 3, 0, 1]]))

        class DeterministicNextTokenModel:
            def eval(self):
                return self

            def __call__(self, input_ids):
                target = (input_ids + 1) % 4
                classes = torch.arange(4).view(1, 1, 4)
                logits = -((classes - target.unsqueeze(-1)) ** 2).float()
                return SimpleNamespace(logits=logits)

        with patch("benchmarks.benchmark_day4_perplexity.log_step"):
            ppl, scored, details = compute_perplexity(
                DeterministicNextTokenModel(), Tokenizer(), "ignored",
                seq_len=4, stride=2, max_eval_tokens=6, device="cpu",
                return_details=True, verify_reference_loss=True,
            )
        # Independently derive the five shifted-label losses over the corpus.
        # Each is scored once despite the overlapping two-window evaluation.
        target_ids = [1, 2, 3, 0, 1]
        expected_mean_nll = sum(
            math.log(sum(math.exp(-((klass - target) ** 2)) for klass in range(4)))
            for target in target_ids
        ) / len(target_ids)
        self.assertEqual(scored, 5)
        self.assertAlmostEqual(details["mean_nll"], expected_mean_nll, places=6)
        self.assertAlmostEqual(ppl, math.exp(expected_mean_nll), places=6)
        self.assertEqual(details["top1_predictions"].tolist(), [1, 2, 3, 0, 1])
        self.assertEqual(details["next_token_accuracy"], 1.0)
        self.assertTrue(details["nll_formula_crosscheck"]["passed"])
        self.assertEqual(details["nll_formula_crosscheck"]["scored_tokens"], 3)

    def test_batched_perplexity_preserves_scores_and_reduces_forward_passes(self):
        class Tokenizer:
            bos_token_id = 0

            def __len__(self):
                return 4

            def __call__(self, text, return_tensors, add_special_tokens=True):
                return SimpleNamespace(input_ids=torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1]]))

        class DeterministicNextTokenModel:
            def eval(self):
                return self

            def __call__(self, input_ids):
                target = (input_ids + 1) % 4
                classes = torch.arange(4).view(1, 1, 4)
                return SimpleNamespace(logits=-((classes - target.unsqueeze(-1)) ** 2).float())

        def evaluate(batch_size):
            with patch("benchmarks.benchmark_day4_perplexity.log_step"):
                return compute_perplexity(
                    DeterministicNextTokenModel(), Tokenizer(), "ignored",
                    seq_len=4, stride=2, max_eval_tokens=10, device="cpu",
                    return_details=True, verify_reference_loss=True,
                    batch_size=batch_size,
                )

        serial_ppl, serial_scored, serial = evaluate(1)
        batched_ppl, batched_scored, batched = evaluate(2)
        self.assertEqual(serial_scored, batched_scored)
        self.assertAlmostEqual(serial_ppl, batched_ppl, places=7)
        self.assertAlmostEqual(serial["mean_nll"], batched["mean_nll"], places=7)
        self.assertEqual(serial["top1_predictions"].tolist(), batched["top1_predictions"].tolist())
        self.assertEqual(serial["token_ids_sha256"], batched["token_ids_sha256"])
        self.assertEqual(serial["window_specification"], batched["window_specification"])
        self.assertEqual(serial["forward_passes"], 4)
        self.assertEqual(batched["forward_passes"], 2)
        self.assertEqual(batched["inference_batch_size"], 2)


if __name__ == "__main__":
    unittest.main()
