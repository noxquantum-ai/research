import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from benchmarks.benchmark_day3_cascade import load_cached_tensor
from benchmarks.benchmark_day4_perplexity import compute_perplexity
from nox.artifacts import checkpoint_file_sizes, checkpoint_storage_bytes, fingerprint
from nox.compiler import estimate_artifact_tensor_bytes
from nox.sequential_compiler import compute_dense_argmax_metrics


class PipelineAccountingTests(unittest.TestCase):
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
            def __call__(self, text, return_tensors):
                return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3, 0, 1]]))

        class UniformModel:
            def eval(self):
                return self

            def __call__(self, input_ids, labels):
                logits = torch.zeros((*input_ids.shape, 4), dtype=torch.float32)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, 4),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                return SimpleNamespace(loss=loss, logits=logits)

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


if __name__ == "__main__":
    unittest.main()
