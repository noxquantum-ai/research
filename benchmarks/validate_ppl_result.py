#!/usr/bin/env python3
"""Validate a completed layer-0 development PPL result before resuming a stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from nox.artifacts import fingerprint
from nox.evaluation import MODEL_ID, MODEL_REVISION, WIKITEXT_REVISION


def validate_result(path: Path, *, variant: str, checkpoint_dir: Path | None = None) -> dict:
    result = json.loads(path.read_text())
    if result.get("status") != "COMPLETE" or result.get("model_variant") != variant:
        raise ValueError("Cached PPL result is incomplete or has the wrong model variant")
    if (result.get("model_id"), result.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("Cached PPL result uses a different model or model revision")
    if result.get("dataset_split") != "validation" or result.get("dataset", {}).get("split") != "validation":
        raise ValueError("Cached result is not the pinned WikiText-2 validation split")
    if result.get("dataset", {}).get("dataset_revision") != WIKITEXT_REVISION:
        raise ValueError("Cached result uses a different WikiText-2 revision")
    if result.get("dataset", {}).get("fallback") is not False:
        raise ValueError("Cached result does not prove use of the canonical dataset source")
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True,
    ).strip()
    if result.get("source_commit") != current_commit:
        raise ValueError("Cached result was produced by a different source commit")
    identity = result.get("protocol_identity", {})
    if result.get("protocol_fingerprint") != fingerprint(identity):
        raise ValueError("Cached PPL protocol fingerprint is invalid")
    expected_protocol = {
        "context_length": 2048,
        "stride": 512,
        "primary_input_tokens": 0,
        "primary_evaluation_scope": "full_split",
        "full_test": False,
        "dtype": "float16",
        "attention_implementation": "sdpa",
    }
    if any(identity.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError("Cached result uses a different development PPL protocol")
    protocol = result.get("protocol", {})
    if (protocol.get("context_length"), protocol.get("stride"), protocol.get("requested_max_input_tokens")) != (
        2048, 512, 0,
    ):
        raise ValueError("Cached result did not score the complete validation split")
    metrics = result.get("metrics", {}).get("primary")
    if not isinstance(metrics, dict):
        raise ValueError("Cached result lacks primary full-split metrics")
    input_tokens = metrics.get("input_tokens")
    scored_tokens = metrics.get("scored_tokens")
    if not isinstance(input_tokens, int) or input_tokens < 10_001 or scored_tokens != input_tokens - 1:
        raise ValueError("Cached result has inconsistent input/target token counts")
    if metrics.get("token_ids_sha256") != protocol.get("token_ids_sha256"):
        raise ValueError("Cached token stream hash is inconsistent")

    sidecars = {}
    for name, dtype in (("token_nll_sidecar", "float32"), ("top1_prediction_sidecar", "int32")):
        metadata = metrics.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(f"Cached result lacks {name}")
        sidecar_path = Path(metadata.get("path", ""))
        if not sidecar_path.is_file():
            raise FileNotFoundError(sidecar_path)
        payload = sidecar_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata.get("sha256"):
            raise ValueError(f"Cached {name} SHA-256 mismatch")
        if metadata.get("count") != scored_tokens:
            raise ValueError(f"Cached {name} count does not match scored tokens")
        if name == "token_nll_sidecar":
            if metadata.get("dtype") != "float32_little_endian_npy":
                raise ValueError("Cached NLL dtype is not float32")
            values = np.load(sidecar_path, allow_pickle=False)
            if values.dtype != np.float32 or values.ndim != 1 or not np.isfinite(values).all():
                raise ValueError("Cached per-token NLL sidecar is malformed")
            if not math.isclose(float(values.mean(dtype=np.float64)), metrics["mean_nll"], rel_tol=2e-6, abs_tol=2e-7):
                raise ValueError("Cached NLL sidecar does not match aggregate mean NLL")
            if not math.isclose(math.exp(float(values.mean(dtype=np.float64))), metrics["perplexity"], rel_tol=2e-6):
                raise ValueError("Cached NLL sidecar does not match perplexity")
        else:
            if metadata.get("dtype") != "int32_little_endian" or len(payload) != 4 * scored_tokens:
                raise ValueError("Cached top-1 prediction sidecar is malformed")
        sidecars[name] = str(sidecar_path)

    if variant == "dense":
        if checkpoint_dir is not None or result.get("compressed_layers") != []:
            raise ValueError("Dense result unexpectedly references NEC checkpoint layers")
    else:
        if checkpoint_dir is None or result.get("compressed_layers") != [0] or not result.get("partial_prefix"):
            raise ValueError("NEC result is not the expected one-layer prefix evaluation")
        resolved = checkpoint_dir.resolve()
        if Path(result.get("checkpoint_dir", "")).resolve() != resolved:
            raise ValueError("Cached NEC result references a different checkpoint directory")
        manifest_path = resolved / "model_manifest.json"
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if result.get("checkpoint_manifest_sha256") != digest:
            raise ValueError("Cached NEC result checkpoint manifest hash is stale")
        audit_path = Path(result.get("checkpoint_audit", ""))
        audit = json.loads(audit_path.read_text())
        if audit.get("status") != "PASS" or audit.get("manifest_sha256") != digest:
            raise ValueError("Cached NEC checkpoint audit does not match its manifest")
    return {
        "status": "PASS",
        "result": str(path.resolve()),
        "variant": variant,
        "source_commit": current_commit,
        "scored_tokens": scored_tokens,
        "perplexity": metrics["perplexity"],
        "sidecars": sidecars,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--variant", choices=("dense", "high_fidelity_nec"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_result(args.result, variant=args.variant, checkpoint_dir=args.checkpoint_dir), indent=2))


if __name__ == "__main__":
    main()
