#!/usr/bin/env python3
"""Full MMLU evaluation; never substitutes fallback questions or partial layers."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints
from nox.artifacts import checkpoint_file_sizes, fingerprint, source_files_fingerprint
from nox.mmlu import load_full_mmlu, prepare_example, score_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--dense-results", type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--max-questions", type=int, default=0, help="Smoke check only; zero runs the full test")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    predictions_path = args.output.with_suffix(".predictions.jsonl")
    if args.output.exists() or predictions_path.exists():
        raise FileExistsError("Use fresh result paths")
    if args.batch_size < 1 or args.max_questions < 0 or not 2 <= args.max_context <= 4096:
        raise ValueError("Invalid evaluation limits")
    model_id, revision = "NousResearch/Llama-2-7b-hf", "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
    splits, data = load_full_mmlu()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    demonstrations = defaultdict(list)
    for row in splits["dev"]:
        demonstrations[row["subject"]].append(row)
    rows = splits["test"][:args.max_questions] if args.max_questions else splits["test"]
    examples = [prepare_example(tokenizer, row, demonstrations[row["subject"]], args.max_context) for row in rows]
    protocol = {"model_id": model_id, "model_revision": revision, "data": data,
                "max_context": args.max_context, "requested_shots": 5,
                "scoring": "single_answer_letter_logit", "questions": len(rows),
                "prompt_hashes": [example["prompt_sha256"] for example in examples],
                "source_fingerprint": source_files_fingerprint([Path(__file__), ROOT / "nox/mmlu.py"])}
    protocol_hash = fingerprint(protocol)
    report = {"status": "PREPARED", "protocol": protocol, "protocol_fingerprint": protocol_hash,
              "full_test": len(rows) == 14042, "evaluation_performed": False,
              "effective_shot_counts": dict(Counter(e["shots"] for e in examples)),
              "input_token_count": sum(len(e["input_ids"]) for e in examples),
              "variant": "compressed" if args.checkpoint_dir else "dense"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
        print(json.dumps({key: report[key] for key in ("status", "full_test", "input_token_count", "effective_shot_counts")}))
        return
    baseline = None
    if args.checkpoint_dir:
        if args.dense_results is None:
            raise ValueError("Compressed evaluation requires matched --dense-results")
        baseline = json.loads(args.dense_results.read_text())
        if baseline.get("status") != "COMPLETE" or baseline.get("variant") != "dense" or baseline["protocol_fingerprint"] != protocol_hash:
            raise ValueError("Dense result is incomplete or uses a different protocol")
        expected = {f"compressed_layer_{index}.pt" for index in range(32)}
        if {path.name for path in args.checkpoint_dir.glob("compressed_layer_*.pt")} != expected:
            raise ValueError("MMLU acceptance evaluation requires all 32 transformer layers")
        manifest = json.loads((args.checkpoint_dir / "model_manifest.json").read_text())
        identity = manifest["run_identity"]
        if identity["model_id"] != model_id or identity["model_revision"] != revision or identity["compression_path"] != "no_healing_independent":
            raise ValueError("Checkpoint model/revision/zero-healing provenance mismatch")
        if fingerprint(identity) != manifest["run_fingerprint"]:
            raise ValueError("Checkpoint manifest fingerprint mismatch")
        for index in range(32):
            checkpoint = torch.load(args.checkpoint_dir / f"compressed_layer_{index}.pt", map_location="cpu", weights_only=True)
            if (checkpoint["layer_idx"] != index or checkpoint.get("run_fingerprint") != manifest["run_fingerprint"]
                    or checkpoint.get("compression_path") != "no_healing_independent"):
                raise ValueError(f"Mixed or stale checkpoint at layer {index}")
            del checkpoint
        report["checkpoint_manifest_fingerprint"] = fingerprint(manifest)
        report["compressed_layer_file_bytes"] = checkpoint_file_sizes(args.checkpoint_dir, range(32))
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16,
                                               device_map={"": "cuda"}, attn_implementation="sdpa").eval()
    if args.checkpoint_dir:
        injected = inject_nox_checkpoints(model, str(args.checkpoint_dir))
        if len(injected) != 32 or set(injected) != set(range(32)):
            raise ValueError("Incomplete transformer replacement")
    totals, correct = Counter(), Counter()
    started = time.monotonic()
    with predictions_path.open("x") as predictions:
        for start in range(0, len(examples), args.batch_size):
            scores = score_batch(model, examples[start:start + args.batch_size], tokenizer.eos_token_id, "cuda")
            for offset, logits in enumerate(scores):
                index = start + offset
                row = rows[index]
                prediction = int(logits.argmax())
                totals[row["subject"]] += 1
                correct[row["subject"]] += int(prediction == row["answer"])
                predictions.write(json.dumps({"index": index, "subject": row["subject"], "label": row["answer"],
                                               "prediction": prediction, "choice_logits": logits.tolist(),
                                               "prompt_sha256": examples[index]["prompt_sha256"]}) + "\n")
            predictions.flush()
            if start % 100 == 0:
                print(f"scored={sum(totals.values())}/{len(rows)} accuracy={sum(correct.values()) / sum(totals.values()):.6f}", flush=True)
    accuracy = sum(correct.values()) / sum(totals.values())
    report.update(status="COMPLETE", evaluation_performed=True, accuracy=accuracy,
                  scored_questions=sum(totals.values()), seconds=time.monotonic() - started,
                  subject_accuracy={subject: correct[subject] / totals[subject] for subject in sorted(totals)},
                  predictions_path=str(predictions_path))
    if baseline:
        if baseline["accuracy"] <= 0:
            raise ValueError("Dense accuracy must be positive for retention")
        report["dense_accuracy_retention"] = accuracy / baseline["accuracy"]
        report["passes_mmlu_gate"] = report["full_test"] and report["dense_accuracy_retention"] >= 0.95
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
    print(f"COMPLETE: {args.output}; full_test={report['full_test']}; accuracy={accuracy:.6f}", flush=True)


if __name__ == "__main__":
    main()
