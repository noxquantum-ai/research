#!/usr/bin/env python3
"""Cache and validate full pinned MMLU inputs; this does not evaluate a model."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    revision = "c30699e8356da336a370243923dbaf21066bb9fe"
    record = {"dataset": "cais/mmlu", "revision": revision, "splits": {},
              "model_id": "NousResearch/Llama-2-7b-hf",
              "model_revision": "8efe6c9b93655b934e27bd9981e3ec13e55aee9d",
              "evaluation_performed": False, "fallback": False}
    for split, expected in (("dev", 285), ("test", 14042)):
        filename = f"all/{split}-00000-of-00001.parquet"
        path = Path(hf_hub_download(record["dataset"], repo_type="dataset", revision=revision, filename=filename))
        rows = pq.read_table(path).to_pylist()
        counts = Counter(row["subject"] for row in rows)
        if len(rows) != expected or len(counts) != 57:
            raise ValueError(f"Incomplete MMLU {split} split")
        if split == "dev" and set(counts.values()) != {5}:
            raise ValueError("Expected five demonstrations per subject")
        if any(len(row["choices"]) != 4 or row["answer"] not in range(4) for row in rows):
            raise ValueError("Invalid question choices or labels")
        record["splits"][split] = {"filename": filename, "rows": len(rows),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "bytes": path.stat().st_size, "subjects": dict(sorted(counts.items()))}
    tokenizer = AutoTokenizer.from_pretrained(record["model_id"], revision=record["model_revision"])
    prompt = "A test question.\nA. One\nB. Two\nC. Three\nD. Four\nAnswer:"
    prefix = tokenizer.encode(prompt, add_special_tokens=True)
    suffixes = {}
    for letter in "ABCD":
        tokens = tokenizer.encode(prompt + " " + letter, add_special_tokens=True)
        if tokens[:len(prefix)] != prefix or len(tokens) != len(prefix) + 1:
            raise ValueError("Answer tokenization requires a different scoring implementation")
        suffixes[letter] = tokens[len(prefix):]
    record["answer_tokenization_probe"] = suffixes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(record, output, indent=2)
    print(json.dumps({"counts": {s: r["rows"] for s, r in record["splits"].items()},
                      "answer_tokens": suffixes, "evaluation_performed": False}))


if __name__ == "__main__":
    main()
