import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.validate_ppl_result import validate_result
from nox.artifacts import fingerprint
from nox.evaluation import MODEL_ID, MODEL_REVISION, WIKITEXT_REVISION


class ValidatePPLResultTests(unittest.TestCase):
    def test_accepts_only_checksum_verified_current_full_validation_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nll = np.resize(np.asarray([1.0, 1.2, 1.1, 1.3], dtype="<f4"), 10_000)
            nll_path = root / "nll.npy"
            with nll_path.open("wb") as stream:
                np.save(stream, nll, allow_pickle=False)
            top1_path = root / "top1.i32"
            top1_path.write_bytes(np.arange(10_000, dtype="<i4").tobytes())
            identity = {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "dataset": {"dataset_revision": WIKITEXT_REVISION, "split": "validation"},
                "context_length": 2048,
                "stride": 512,
                "primary_input_tokens": 0,
                "primary_evaluation_scope": "full_split",
                "full_test": False,
                "dtype": "float16",
                "attention_implementation": "sdpa",
            }
            result = {
                "status": "COMPLETE",
                "model_variant": "dense",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "source_commit": __import__("subprocess").check_output(
                    ["git", "rev-parse", "HEAD"], text=True,
                ).strip(),
                "dataset_split": "validation",
                "dataset": {
                    "split": "validation", "dataset_revision": WIKITEXT_REVISION,
                    "source_sha256": "source-hash", "source_url": "pinned-url", "fallback": False,
                },
                "protocol_identity": identity,
                "protocol_fingerprint": fingerprint(identity),
                "protocol": {
                    "context_length": 2048, "stride": 512, "requested_max_input_tokens": 0,
                    "token_ids_sha256": "token-hash",
                },
                "compressed_layers": [],
                "metrics": {"primary": {
                    "input_tokens": 10_001,
                    "scored_tokens": 10_000,
                    "token_ids_sha256": "token-hash",
                    "mean_nll": float(nll.mean(dtype=np.float64)),
                    "perplexity": float(np.exp(nll.mean(dtype=np.float64))),
                    "token_nll_sidecar": {
                        "path": str(nll_path), "sha256": hashlib.sha256(nll_path.read_bytes()).hexdigest(),
                        "dtype": "float32_little_endian_npy", "count": 10_000,
                    },
                    "top1_prediction_sidecar": {
                        "path": str(top1_path), "sha256": hashlib.sha256(top1_path.read_bytes()).hexdigest(),
                        "dtype": "int32_little_endian", "count": 10_000,
                    },
                }},
            }
            result_path = root / "dense.json"
            result_path.write_text(json.dumps(result))
            self.assertEqual(validate_result(result_path, variant="dense")["status"], "PASS")
            result["source_commit"] = "old-commit"
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "different source commit"):
                validate_result(result_path, variant="dense")


if __name__ == "__main__":
    unittest.main()
