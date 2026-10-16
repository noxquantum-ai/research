#!/usr/bin/env python3
"""
Updates README.md and Novelty.md with the final multi-tier results and Day 5 benchmarks.
"""

import json
import os
import sys
from pathlib import Path


def main():
    root_dir = Path(__file__).resolve().parent.parent
    sweep_dir = root_dir / "checkpoints" / "day3_day4_cascade" / "master_cliff_sweep"
    day5_file = sweep_dir / "day5_multitask_results.json"

    print(f"Reading benchmark results from '{sweep_dir}'...")

    tier_metrics = {}
    for tier in ["extreme_77x", "production_18x", "high_fidelity_10x"]:
        m_file = sweep_dir / tier / "model_manifest.json"
        if m_file.exists():
            with open(m_file, "r", encoding="utf-8") as f:
                tier_metrics[tier] = json.load(f)

    day5_results = {}
    if day5_file.exists():
        with open(day5_file, "r", encoding="utf-8") as f:
            day5_results = json.load(f)

    # Generate benchmark summary markdown block
    summary_lines = [
        "## Recorded Multi-Tier Results (Full 32-Layer LLaMA-2 7B)",
        "",
        "Checkpoint size is the serialized layer-checkpoint bundle; it excludes unchanged base-model tensors. The bundle alone is not a full-model compression ratio.",
        "",
        "Only benchmark results with recorded canonical dataset provenance are shown as scores; curated fallback pools are marked unavailable.",
        "",
        "| Architecture / Model | Serialized checkpoint bundle | Full-model ratio | WikiText-2 PPL | MMLU (%) | GSM8K (%) | HellaSwag (%) | BoolQ (%) | ARC-C (%) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    dense_res = day5_results.get("dense_fp16", day5_results.get("external_dense_reference", {}))
    dense_measured = dense_res.get("measured_by_this_run", "Official" not in dense_res.get("tier", ""))
    dense_label = (
        "Dense baseline measured in this run"
        if dense_measured
        else "Published dense reference (not measured here; protocol differs)"
    )
    summary_lines.append(
        f"| **{dense_label}** | N/A | N/A | {dense_res.get('wikitext2_ppl', 'N/A')} | {dense_res.get('mmlu_acc', 'N/A')} | {dense_res.get('gsm8k_acc', 'N/A')} | {dense_res.get('hellaswag_acc', 'N/A')} | {dense_res.get('boolq_acc', 'N/A')} | {dense_res.get('arc_challenge_acc', 'N/A')} |"
    )

    tier_order = [
        ("high_fidelity_10x", "Nox High-Fidelity"),
        ("production_18x", "Nox Production"),
        ("extreme_77x", "Nox Extreme"),
    ]

    for t_id, t_name in tier_order:
        manifest = tier_metrics.get(t_id, {})
        d5 = day5_results.get(t_id, {})

        stored_bytes = manifest.get("serialized_checkpoint_bytes")
        candidate_dirs = [sweep_dir / t_id, root_dir / "compressed_models" / "tiers" / t_id]
        if stored_bytes is None:
            for candidate_dir in candidate_dirs:
                files = list(candidate_dir.glob("compressed_layer_*.pt")) if candidate_dir.exists() else []
                if files:
                    stored_bytes = sum(p.stat().st_size for p in files)
                    break
        actual_size = f"{stored_bytes / (1024.0 * 1024.0):.1f} MiB" if stored_bytes else "not recorded"

        provenance = d5.get("dataset_provenance", {})
        w_source = provenance.get("wikitext2", {})
        w_ppl = d5.get("wikitext2_ppl", "N/A") if w_source and not w_source.get("fallback") else "N/A (provenance missing)"

        def verified_accuracy(key, dataset_key):
            dataset_meta = provenance.get(dataset_key, {})
            if dataset_meta and not dataset_meta.get("fallback"):
                return d5.get(key, "N/A")
            return "N/A (curated fallback or provenance missing)"

        mmlu = verified_accuracy("mmlu_acc", "mmlu")
        gsm = verified_accuracy("gsm8k_acc", "gsm8k")
        hs = verified_accuracy("hellaswag_acc", "hellaswag")
        boolq = verified_accuracy("boolq_acc", "boolq")
        arc = verified_accuracy("arc_challenge_acc", "arc_challenge")

        summary_lines.append(
            f"| **{t_name}** | **{actual_size}** | N/A (base tensors excluded) | {w_ppl} | {mmlu} | {gsm} | {hs} | {boolq} | {arc} |"
        )

    summary_block = "\n".join(summary_lines) + "\n\n"
    print("Generated Summary Block:\n" + summary_block)

    # Append or update in README.md
    readme_path = root_dir / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
        section_header = "## Verified Multi-Tier Empirical Benchmark Scorecard"
        if section_header in content:
            pre, post = content.split(section_header, 1)
            # Find next header or end
            next_header = post.find("\n## ")
            if next_header != -1:
                content = pre + summary_block + post[next_header + 1:]
            else:
                content = pre + summary_block
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("Updated existing benchmark scorecard in README.md.")
        else:
            with open(readme_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + summary_block)
            print("Appended results to README.md.")

    # Write standalone report
    report_path = root_dir / "FINAL_MULTI_TIER_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# NoxQuantum Final Empirical Benchmark Report\n\n" + summary_block)
    print(f"Wrote final report to '{report_path}'.")


if __name__ == "__main__":
    main()
