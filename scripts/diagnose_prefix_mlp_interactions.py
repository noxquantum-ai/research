#!/usr/bin/env python3
"""Factorial saved-weight MLP controls and exact SwiGLU error decomposition."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_prefix_failure import PROJECTIONS, NORMS, module, install, capture, columns, summarize, relative
from scripts.diagnose_cross_layer_interactions import forward_details
from benchmarks.benchmark_day4_perplexity import compute_perplexity, inject_nox_checkpoints
from benchmarks.evaluate_high_fidelity_nec import load_model
from nox.data import load_wikitext2_raw_split
from nox.evaluation import sha256_file, MODEL_ID, MODEL_REVISION


@torch.inference_mode()
def mlp_error_terms(dense, compressed, x):
    g, u = dense["gate_proj"] @ x, dense["up_proj"] @ x
    gh, uh = compressed["gate_proj"] @ x, compressed["up_proj"] @ x
    h, hh = F.silu(g) * u, F.silu(gh) * uh
    d, dh = dense["down_proj"], compressed["down_proj"]
    y, yh = d @ h, dh @ hh
    product = d @ (hh - h)
    down = (dh - d) @ h
    cross = (dh - d) @ (hh - h)
    denominator = y.norm().clamp_min(1e-30)
    delta_silu, delta_up = F.silu(gh) - F.silu(g), uh - u
    return {
        "gate_output": summarize(g.T, gh.T), "up_output": summarize(u.T, uh.T),
        "swiglu_product": summarize(h.T, hh.T), "mlp_output": summarize(y.T, yh.T),
        "product_error_through_dense_down_relative": float(product.norm() / denominator),
        "down_error_on_dense_product_relative": float(down.norm() / denominator),
        "down_error_on_changed_product_cross_term_relative": float(cross.norm() / denominator),
        "down_error_on_actual_product_relative": float((down + cross).norm() / denominator),
        "product_down_error_cosine": float(F.cosine_similarity(product.flatten(), (down + cross).flatten(), dim=0)),
        "output_decomposition_residual": float(((yh - y) - product - down - cross).norm() / denominator),
        "product_decomposition_residual": float(((hh - h) - delta_silu * u - F.silu(g) * delta_up - delta_silu * delta_up).norm() / h.norm().clamp_min(1e-30)),
    }


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-ppl-report", type=Path,
                    help="Reuse fully completed, matching PPL controls from a prior diagnostic report.")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.monotonic()
    manifest_path = args.checkpoint_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {f"compressed_layer_{i}.pt": sha256_file(args.checkpoint_dir / f"compressed_layer_{i}.pt") for i in range(4)}
    assert hashes == manifest["checkpoint_file_sha256"]
    report = {"status": "RUNNING", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "script_sha256": sha256_file(__file__), "helper_sha256": sha256_file(ROOT / "scripts/diagnose_prefix_failure.py"),
              "checkpoint_manifest_sha256": sha256_file(manifest_path), "checkpoint_sha256": hashes,
              "scope": "diagnostic saved-weight interventions, no fitting",
              "protocol": {"split": "validation", "max_eval_tokens": 4096, "context": 2048, "stride": 512},
              "ppl": {}, "decomposition": {}}

    def save():
        report["elapsed_seconds"] = time.monotonic() - start
        tmp = args.output / "report.partial.json"
        tmp.write_text(json.dumps(report, indent=2) + "\n")
        tmp.replace(args.output / "report.json")

    print("PHASE=load_dense", flush=True)
    model, tokenizer, _, _, _ = load_model(None, variant="dense", offload_folder=args.output / "offload")
    dense = {i: {n: module(model.model.layers[i], n).weight.detach().cpu().clone() for n in (*PROJECTIONS, *NORMS)} for i in range(4)}
    inject_nox_checkpoints(model, str(args.checkpoint_dir), expected_layers=list(range(4)))
    compressed = {i: {n: module(model.model.layers[i], n).weight.detach().cpu().clone() for n in (*PROJECTIONS, *NORMS)} for i in range(4)}
    text, report["dataset"] = load_wikitext2_raw_split("validation")
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    groups = {"dense": (), "gate": ("gate_proj",), "up": ("up_proj",), "down": ("down_proj",),
              "gate_up": ("gate_proj", "up_proj"), "gate_down": ("gate_proj", "down_proj"),
              "up_down": ("up_proj", "down_proj"), "mlp": PROJECTIONS[4:], "full_nec": PROJECTIONS}
    if args.reuse_ppl_report:
        previous_path = args.reuse_ppl_report
        previous = json.loads(previous_path.read_text())
        expected_protocol = {"split": "validation", "max_eval_tokens": 4096, "context": 2048, "stride": 512}
        if (previous.get("model_id"), previous.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("PPL reuse report has different model provenance")
        if previous.get("checkpoint_manifest_sha256") != report["checkpoint_manifest_sha256"]:
            raise ValueError("PPL reuse report belongs to a different checkpoint manifest")
        if previous.get("checkpoint_sha256") != report["checkpoint_sha256"]:
            raise ValueError("PPL reuse report belongs to different checkpoint files")
        if previous.get("protocol") != expected_protocol or set(previous.get("ppl", {})) != set(groups):
            raise ValueError("PPL reuse report lacks the complete matching control set")
        if any(row.get("scored_tokens") != 4095 or not row.get("nll_formula_crosscheck", {}).get("passed")
               for row in previous["ppl"].values()):
            raise ValueError("PPL reuse report includes incomplete or unchecked measurements")
        report["ppl"] = previous["ppl"]
        report["ppl_reused_from"] = {"path": str(previous_path),
                                     "sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest()}
        print(f"REUSED_COMPLETE_PPL report={previous_path}", flush=True)
    else:
        for name, group in groups.items():
            install(model, dense, compressed, {i: group for i in range(4)})
            ppl, count, details = compute_perplexity(model, tokenizer, text, max_eval_tokens=4096,
                return_details=True, verify_reference_loss=True, return_token_nll=True, progress_interval=1)
            torch.save({"token_nll": details.pop("token_nll"), "predictions": details.pop("top1_predictions")}, args.output / f"{name}.pt")
            report["ppl"][name] = {"perplexity": ppl, "mean_nll": details["mean_nll"],
                "next_token_accuracy": details["next_token_accuracy"], "scored_tokens": count,
                "compressed_projections_each_layer": list(group), "nll_formula_crosscheck": details.get("nll_formula_crosscheck")}
            print(f"RESULT {name} {json.dumps(report['ppl'][name])}", flush=True)
            save()
    offsets = manifest["run_identity"]["input_data"]["validation_window_offsets"]
    saved_ids = torch.stack([ids[0, j:j+128] for j in offsets])
    for scope, sample in (("saved_validation_128", saved_ids), ("first2048_long", ids[:, :2048])):
        install(model, dense, compressed, {})
        clean = capture(model, sample)
        install(model, dense, compressed, {i: PROJECTIONS for i in range(4)})
        changed = capture(model, sample)
        report["decomposition"][scope] = {}
        for i in range(4):
            wd = {k: v.to(model.device, torch.float32) for k, v in dense[i].items()}
            wc = {k: v.to(model.device, torch.float32) for k, v in compressed[i].items()}
            opts = dict(n_heads=model.config.num_attention_heads, n_kv_heads=model.config.num_key_value_heads,
                        seq_len=sample.shape[1], rms_norm_eps=model.config.rms_norm_eps)
            dd = forward_details(wd, columns(clean[i]["input"], model.device), **opts)
            cc = forward_details(wc, columns(changed[i]["input"], model.device), **opts)
            x, xh = dd["mlp_input"], cc["mlp_input"]
            r = {"custom_dense_vs_hf": summarize(clean[i]["output"], dd["output"].T),
                 "custom_compressed_vs_hf": summarize(changed[i]["output"], cc["output"].T),
                 "normalized_mlp_input_disagreement": summarize(x.T, xh.T),
                 "dense_context": mlp_error_terms(wd, wc, x),
                 "compressed_context": mlp_error_terms(wd, wc, xh),
                 "actual_cascade_mlp_error": summarize(dd["mlp_output"].T, cc["mlp_output"].T)}
            report["decomposition"][scope][str(i)] = r
            print(f"DECOMPOSITION scope={scope} layer={i} {json.dumps({k:r['dense_context'][k] for k in ['product_error_through_dense_down_relative','down_error_on_dense_product_relative','down_error_on_changed_product_cross_term_relative']})}", flush=True)
            del wd, wc, dd, cc, x, xh
            save()
    report["status"] = "COMPLETE"
    report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save()
    print(f"COMPLETE elapsed={report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
