import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.compare_ppl_results import compare_results


class PPLComparisonTests(unittest.TestCase):
    def test_paired_comparison_uses_verified_nll_and_prediction_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dense_nll = np.full(1024, 1.5, dtype=np.float32)
            nec_nll = dense_nll + np.linspace(0.01, 0.03, 1024, dtype=np.float32)
            dense_top1 = np.arange(1024, dtype="<i4")
            nec_top1 = dense_top1.copy()
            nec_top1[::10] += 1

            def write_nll(name, values):
                path = root / f"{name}.npy"
                with path.open("wb") as stream:
                    np.save(stream, values, allow_pickle=False)
                payload = path.read_bytes()
                return {
                    "path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "count": values.size,
                    "dtype": "float32_little_endian_npy",
                }

            def write_top1(name, values):
                path = root / f"{name}.i32"
                payload = values.tobytes()
                path.write_bytes(payload)
                return {
                    "path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "count": values.size,
                    "dtype": "int32_little_endian",
                }

            common = {
                "status": "COMPLETE",
                "model_id": "NousResearch/Llama-2-7b-hf",
                "model_revision": "pinned-revision",
                "source_commit": "abc123",
                "dataset_split": "validation",
                "protocol_fingerprint": "same-protocol",
                "dataset": {"dataset_revision": "dataset-sha", "split": "validation"},
            }

            def result(variant, nll, top1, accuracy):
                return {
                    **common,
                    "model_variant": variant,
                    "compressed_layers": [] if variant == "dense" else [0],
                    "metrics": {"primary_10k": {
                        "input_tokens": 1025,
                        "scored_tokens": 1024,
                        "token_ids_sha256": "same-token-stream",
                        "window_count": 3,
                        "perplexity": float(np.exp(nll.mean())),
                        "mean_nll": float(nll.mean()),
                        "next_token_accuracy": accuracy,
                        "token_nll_sidecar": write_nll(variant + "-nll", nll),
                        "top1_prediction_sidecar": write_top1(variant + "-top1", top1),
                    }},
                }

            report = compare_results(
                result("dense", dense_nll, dense_top1, 0.4),
                result("high_fidelity_nec", nec_nll, nec_top1, 0.39),
                repetitions=500,
                seed=7,
            )
            self.assertAlmostEqual(report["paired"]["mean_nll_delta_nec_minus_dense"], 0.02, places=6)
            self.assertAlmostEqual(report["paired"]["top1_agreement_with_dense"], 921 / 1024, places=6)
            self.assertEqual(report["paired"]["scored_tokens"], 1024)

    def test_rejects_different_token_protocols(self):
        dense = {"status": "COMPLETE", "model_variant": "dense"}
        nec = {"status": "COMPLETE", "model_variant": "high_fidelity_nec"}
        with self.assertRaisesRegex(ValueError, "model_id"):
            compare_results(dense, nec)


if __name__ == "__main__":
    unittest.main()
