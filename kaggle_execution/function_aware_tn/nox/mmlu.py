"""Pinned full-test MMLU data and explicit answer-letter evaluation helpers."""
import hashlib
from collections import Counter
import math
from pathlib import Path

import torch

MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def resolve_mmlu_reference(dense_result=None, reference_accuracy=None,
                           reference_source=None, protocol_fingerprint=None):
    """Validate either a matched dense run or an explicitly external score."""
    if dense_result is not None and reference_accuracy is not None:
        raise ValueError("Choose either a matched dense result or an external reference")
    if dense_result is not None:
        if reference_source:
            raise ValueError("--reference-source applies only to an external reference")
        if (dense_result.get("status") != "COMPLETE"
                or dense_result.get("variant") != "dense"
                or dense_result.get("protocol_fingerprint") != protocol_fingerprint):
            raise ValueError("Dense result is incomplete or uses a different protocol")
        accuracy = dense_result.get("accuracy")
        if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 < accuracy <= 1:
            raise ValueError("Dense result has an invalid accuracy")
        return {
            "kind": "matched_dense_run",
            "accuracy": float(accuracy),
            "protocol_match": True,
            "source": "dense_results",
        }
    if reference_accuracy is None:
        raise ValueError("Compressed MMLU evaluation requires a reference accuracy")
    if (not isinstance(reference_accuracy, (int, float))
            or not math.isfinite(reference_accuracy) or not 0 < reference_accuracy <= 1):
        raise ValueError("External reference accuracy must be between 0 and 1")
    if not reference_source or not reference_source.strip():
        raise ValueError("External reference requires a nonempty source")
    return {
        "kind": "published_external_score",
        "accuracy": float(reference_accuracy),
        "protocol_match": False,
        "source": reference_source.strip(),
    }


def compare_mmlu_reference(accuracy, reference, full_test, minimum_retention=0.95):
    """Report overall full-test accuracy retention without mixing reference types."""
    if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("MMLU accuracy must be between 0 and 1")
    if not 0 < minimum_retention <= 1:
        raise ValueError("minimum_retention must be between 0 and 1")
    reference_accuracy = reference["accuracy"]
    ratio = accuracy / reference_accuracy
    if reference["protocol_match"]:
        return {
            "dense_accuracy_retention": ratio,
            "passes_mmlu_gate": bool(full_test and ratio + 1e-12 >= minimum_retention),
        }
    threshold = minimum_retention * reference_accuracy
    return {
        "published_reference_accuracy_ratio": ratio,
        "published_reference_95pct_threshold": threshold,
        "passes_published_reference_threshold": bool(full_test and accuracy + 1e-12 >= threshold),
    }


def load_full_mmlu():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    splits, metadata = {}, {"dataset": "cais/mmlu", "revision": MMLU_REVISION, "splits": {}}
    for split, count in (("dev", 285), ("test", 14042)):
        path = Path(hf_hub_download("cais/mmlu", repo_type="dataset", revision=MMLU_REVISION,
                                   filename=f"all/{split}-00000-of-00001.parquet"))
        rows = pq.read_table(path).to_pylist()
        subjects = Counter(row["subject"] for row in rows)
        if len(rows) != count or len(subjects) != 57:
            raise ValueError(f"Incomplete MMLU {split}")
        if split == "dev" and set(subjects.values()) != {5}:
            raise ValueError("Expected five demonstrations per subject")
        if any(len(row["choices"]) != 4 or row["answer"] not in range(4) for row in rows):
            raise ValueError("Invalid MMLU choices or labels")
        splits[split] = rows
        metadata["splits"][split] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "bytes": path.stat().st_size, "subjects": dict(sorted(subjects.items()))}
    return splits, metadata


def format_question(row, include_answer=False):
    text = row["question"] + "".join(f"\n{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
    text += "\nAnswer:"
    return text + f" {'ABCD'[row['answer']]}\n\n" if include_answer else text


def prepare_example(tokenizer, row, demonstrations, max_context=4096):
    if len(demonstrations) != 5 or any(d["subject"] != row["subject"] for d in demonstrations):
        raise ValueError("Expected five demonstrations from the same subject")
    # The reference evaluator includes a leading space in format_subject.
    header = "The following are multiple choice questions (with answers) about  " + row["subject"].replace("_", " ") + ".\n\n"
    for shots in range(5, -1, -1):
        prompt = header + "".join(format_question(d, True) for d in demonstrations[:shots]) + format_question(row)
        ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(ids) + 1 <= max_context:
            break
    else:
        raise ValueError("Question exceeds context even with no demonstrations; refusing truncation")
    choices = []
    for letter in "ABCD":
        appended = tokenizer.encode(prompt + " " + letter, add_special_tokens=True)
        if appended[:len(ids)] != ids or len(appended) != len(ids) + 1:
            raise ValueError("Answer is not a single stable continuation token")
        choices.append(appended[-1])
    if len(set(choices)) != 4:
        raise ValueError("Answer token IDs must be distinct")
    return {"input_ids": ids, "choice_token_ids": choices, "shots": shots,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


@torch.inference_mode()
def score_batch(model, examples, pad_token_id, device):
    length = max(len(example["input_ids"]) for example in examples)
    ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, example in enumerate(examples):
        tokens = example["input_ids"]
        ids[index, :len(tokens)] = torch.tensor(tokens, device=device)
        mask[index, :len(tokens)] = 1
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    last = logits[torch.arange(len(examples), device=device), mask.sum(dim=1) - 1].float()
    choices = torch.tensor([example["choice_token_ids"] for example in examples], device=device)
    return last.gather(1, choices).cpu()
