"""Pinned full-test MMLU data and explicit answer-letter evaluation helpers."""
import hashlib
from collections import Counter
from pathlib import Path

import torch

MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


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
