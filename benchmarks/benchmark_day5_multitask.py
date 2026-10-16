"""
Day 5 Benchmark: Multi-Task Real-World Evaluation Suite.

Evaluates Nox-compressed checkpoints across standardized language and reasoning tasks:
1. WikiText-2 Perplexity (Language Modeling)
2. C4 Perplexity (Real-world web text modeling)
3. ARC-Challenge (Science Reasoning - Multiple Choice Log-Likelihood)
4. HellaSwag (Commonsense Completion - Multiple Choice Log-Likelihood)
5. MMLU (Academic Multitask Knowledge - 57 subjects)
6. GSM8K (Math Word Problem Reasoning - Exact Match Number Generation)

Demonstrates whether the mathematical error bounds of Nox translate to real-world tasks
without catastrophic reasoning collapse.
"""

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.benchmark_day4_perplexity import (
    compute_perplexity,
    inject_nox_checkpoints,
    load_wikitext2_test_text,
)
from nox.artifacts import checkpoint_storage_bytes

DATASET_PROVENANCE = {}


def log_step(msg):
    t_str = time.strftime("%H:%M:%S")
    print(f"[{t_str}] [Day 5] {msg}", flush=True)


def load_c4_validation_text(max_tokens=10000):
    """
    Loads C4 validation text using datasets or raw fallback.
    """
    try:
        from datasets import load_dataset
        log_step("Loading C4 validation set via datasets...")
        ds = load_dataset("c4", "en", split="validation", streaming=True)
        text_samples = []
        for sample in ds:
            text_samples.append(sample["text"])
            if len(" ".join(text_samples).split()) >= max_tokens * 2:
                break
        DATASET_PROVENANCE["c4"] = {
            "source": "Hugging Face c4/en validation",
            "fallback": False,
        }
        return "\n\n".join(text_samples)
    except Exception as e:
        log_step(f"datasets C4 load failed ({e}), using diverse raw web samples...")
        DATASET_PROVENANCE["c4"] = {
            "source": "handwritten curated text",
            "fallback": True,
            "reason": str(e),
        }
        # Curated diverse web text samples
        sample_texts = [
            "Quantum entanglement is a physical phenomenon that occurs when a group of particles are generated, "
            "interact, or share spatial proximity in ways such that the quantum state of each particle cannot be "
            "described independently of the state of the others. Matrix Product Operators (MPO) and Tensor Trains "
            "provide low-rank tensor decompositions representing 1D and 2D lattice Hamiltonians with polynomial scaling.",
            "In machine learning, transformer architectures rely on self-attention mechanisms to dynamically route "
            "information across sequence positions. Weight matrix factorization compresses the linear projections "
            "while causal healing corrects intra-block and inter-layer residual drifts.",
            "The economics of renewable energy transition depends on capital expenditure reductions in solar photovoltaic "
            "modules, lithium-ion battery energy storage systems, and transmission grid infrastructure modernization.",
            "The immune system comprises innate and adaptive components designed to recognize and neutralize pathogenic "
            "organisms while maintaining self-tolerance through central and peripheral regulatory mechanisms."
        ] * 40
        return "\n\n".join(sample_texts)


@torch.no_grad()
def evaluate_arc_challenge(model, tokenizer, num_samples=100, device="cuda"):
    """
    Evaluates zero-shot multiple-choice accuracy on AI2 Reasoning Challenge (ARC-Challenge).
    """
    log_step(f"Evaluating ARC-Challenge ({num_samples} questions)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        samples = list(ds)[:num_samples]
        DATASET_PROVENANCE["arc_challenge"] = {
            "source": "Hugging Face ai2_arc/ARC-Challenge test",
            "fallback": False,
            "examples": len(samples),
        }
    except Exception as e:
        log_step(f"datasets ARC load failed ({e}), using curated ARC sample pool...")
        DATASET_PROVENANCE["arc_challenge"] = {
            "source": "handwritten curated questions",
            "fallback": True,
            "reason": str(e),
        }
        samples = [
            {
                "question": "Which property of a mineral can be determined just by looking at it?",
                "choices": {"text": ["luster", "hardness", "weight", "magnetism"], "label": ["A", "B", "C", "D"]},
                "answerKey": "A",
            },
            {
                "question": "Which of the following describes a physical change?",
                "choices": {"text": ["iron rusting", "wood burning", "ice melting", "milk souring"], "label": ["A", "B", "C", "D"]},
                "answerKey": "C",
            },
            {
                "question": "What primary force holds the planets in orbit around the Sun?",
                "choices": {"text": ["friction", "gravity", "magnetism", "radiation"], "label": ["A", "B", "C", "D"]},
                "answerKey": "B",
            },
            {
                "question": "Which organelle is known as the powerhouse of eukaryotic cells?",
                "choices": {"text": ["nucleus", "ribosome", "mitochondrion", "golgi apparatus"], "label": ["A", "B", "C", "D"]},
                "answerKey": "C",
            },
        ] * (num_samples // 4 + 1)
        samples = samples[:num_samples]

    correct = 0
    total = 0

    for item in samples:
        q_text = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        gold_key = item["answerKey"]

        # Find log-likelihood of each choice continuation
        choice_scores = []
        for choice in choices:
            prompt = f"Question: {q_text}\nAnswer:"
            full_text = f"{prompt} {choice}"
            prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            full_tokens = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

            prefix_len = prompt_tokens.shape[1]
            if full_tokens.shape[1] <= prefix_len:
                choice_scores.append(-1e9)
                continue

            with torch.no_grad():
                logits = model(full_tokens).logits
                # Target tokens are after prefix
                target_ids = full_tokens[:, prefix_len:].contiguous()
                target_logits = logits[:, prefix_len - 1 : -1, :].contiguous()
                log_probs = F.log_softmax(target_logits, dim=-1)
                choice_ll = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).sum().item()
                # Length normalization
                choice_scores.append(choice_ll / max(1, target_ids.shape[1]))

        pred_idx = int(torch.tensor(choice_scores).argmax().item())
        pred_label = labels[pred_idx] if pred_idx < len(labels) else "A"
        if pred_label == gold_key or (gold_key in "1234" and str(pred_idx + 1) == gold_key):
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    log_step(f"ARC-Challenge Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc, total


@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, num_samples=100, device="cuda"):
    """
    Evaluates zero-shot commonsense reasoning accuracy on HellaSwag.
    """
    log_step(f"Evaluating HellaSwag ({num_samples} questions)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("hellaswag", split="validation")
        samples = list(ds)[:num_samples]
        DATASET_PROVENANCE["hellaswag"] = {
            "source": "Hugging Face hellaswag validation",
            "fallback": False,
            "examples": len(samples),
        }
    except Exception as e:
        log_step(f"datasets HellaSwag load failed ({e}), using curated HellaSwag sample pool...")
        DATASET_PROVENANCE["hellaswag"] = {
            "source": "handwritten curated questions",
            "fallback": True,
            "reason": str(e),
        }
        samples = [
            {
                "ctx": "A person is slicing an apple on a wooden cutting board with a sharp knife.",
                "endings": [
                    "Then, they arrange the slices neatly on a porcelain plate.",
                    "Then, the knife transforms into a flying bird.",
                    "Then, they jump out the window into a swimming pool.",
                    "Then, the apple turns into a brick of solid gold.",
                ],
                "label": "0",
            },
            {
                "ctx": "A musician is tuning a guitar by tightening the tuning pegs.",
                "endings": [
                    "They strike each string until the pitch matches the digital tuner.",
                    "They smash the guitar with a hammer to make it louder.",
                    "The strings begin singing opera by themselves.",
                    "The guitar dissolves into water on the floor.",
                ],
                "label": "0",
            },
        ] * (num_samples // 2 + 1)
        samples = samples[:num_samples]

    correct = 0
    total = 0

    for item in samples:
        ctx = item.get("ctx", "") or item.get("ctx_a", "") + " " + item.get("ctx_b", "")
        endings = item["endings"]
        gold_label = int(item["label"])

        choice_scores = []
        for ending in endings:
            full_text = f"{ctx.strip()} {ending.strip()}"
            ctx_tokens = tokenizer(ctx, return_tensors="pt").input_ids.to(device)
            full_tokens = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

            prefix_len = ctx_tokens.shape[1]
            if full_tokens.shape[1] <= prefix_len:
                choice_scores.append(-1e9)
                continue

            with torch.no_grad():
                logits = model(full_tokens).logits
                target_ids = full_tokens[:, prefix_len:].contiguous()
                target_logits = logits[:, prefix_len - 1 : -1, :].contiguous()
                log_probs = F.log_softmax(target_logits, dim=-1)
                choice_ll = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).sum().item()
                choice_scores.append(choice_ll / max(1, target_ids.shape[1]))

        pred_idx = int(torch.tensor(choice_scores).argmax().item())
        if pred_idx == gold_label:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    log_step(f"HellaSwag Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc, total


@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, num_samples=30, max_new_tokens=96, device="cuda"):
    """
    Evaluates multi-step mathematical reasoning on GSM8K word problems.
    """
    log_step(f"Evaluating GSM8K Math Reasoning ({num_samples} questions)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        samples = list(ds)[:num_samples]
        DATASET_PROVENANCE["gsm8k"] = {
            "source": "Hugging Face gsm8k/main test",
            "fallback": False,
            "examples": len(samples),
        }
    except Exception as e:
        log_step(f"datasets GSM8K load failed ({e}), using curated math sample pool...")
        DATASET_PROVENANCE["gsm8k"] = {
            "source": "handwritten curated questions",
            "fallback": True,
            "reason": str(e),
        }
        samples = [
            {
                "question": "Janet has 3 cats. Each cat eats 2 cans of food per day. How many cans do they eat in a week?",
                "answer": "3 cats * 2 cans/day = 6 cans/day. 6 cans/day * 7 days = 42. #### 42",
            },
            {
                "question": "A baker bakes 50 loaves of bread. He sells 35 in the morning and 10 in the afternoon. How many loaves are left?",
                "answer": "50 - 35 = 15. 15 - 10 = 5. #### 5",
            },
            {
                "question": "Tom buys 4 books at $12 each. If he pays with a $50 bill, how much change does he receive?",
                "answer": "4 * 12 = 48. 50 - 48 = 2. #### 2",
            },
        ] * (num_samples // 3 + 1)
        samples = samples[:num_samples]

    correct = 0
    total = 0

    for item in samples:
        q_text = item["question"]
        gold_ans = item["answer"].split("####")[-1].strip()

        prompt = f"Question: {q_text}\nAnswer with step-by-step reasoning and conclude with the number: #### "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        # Extract first numeric sequence in output
        found_nums = re.findall(r"[-+]?\d*\.?\d+", gen_text)
        gold_num = re.findall(r"[-+]?\d*\.?\d+", gold_ans)

        if found_nums and gold_num and found_nums[0] == gold_num[0]:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    log_step(f"GSM8K Math Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc, total


@torch.no_grad()
def evaluate_mmlu(model, tokenizer, num_samples=60, device="cuda"):
    """
    Evaluates zero-shot / few-shot accuracy on MMLU (Massive Multitask Language Understanding).
    Covers humanities, STEM, social sciences, and professional topics.
    """
    log_step(f"Evaluating MMLU Multitask Knowledge ({num_samples} questions)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        samples = list(ds)[:num_samples]
        DATASET_PROVENANCE["mmlu"] = {
            "source": "Hugging Face cais/mmlu all test",
            "fallback": False,
            "examples": len(samples),
        }
    except Exception as e:
        log_step(f"datasets MMLU load failed ({e}), using curated MMLU benchmark pool...")
        DATASET_PROVENANCE["mmlu"] = {
            "source": "handwritten curated questions",
            "fallback": True,
            "reason": str(e),
        }
        samples = [
            {
                "question": "Which of the following is true of the velocity of an object in uniform circular motion?",
                "choices": ["It is constant in magnitude and direction.", "It is constant in magnitude only.", "It is constant in direction only.", "It is variable in both magnitude and direction."],
                "answer": 1,
            },
            {
                "question": "Which of the following constitutional amendments abolished slavery in the United States?",
                "choices": ["Tenth Amendment", "Twelfth Amendment", "Thirteenth Amendment", "Fourteenth Amendment"],
                "answer": 2,
            },
            {
                "question": "In microeconomics, a firm maximizes profit by producing where:",
                "choices": ["Marginal Revenue equals Marginal Cost", "Total Revenue equals Total Cost", "Average Variable Cost is minimized", "Price equals Average Fixed Cost"],
                "answer": 0,
            },
            {
                "question": "Which sorting algorithm exhibits an average-case time complexity of O(N log N)?",
                "choices": ["Bubble Sort", "Merge Sort", "Insertion Sort", "Selection Sort"],
                "answer": 1,
            },
        ] * (num_samples // 4 + 1)
        samples = samples[:num_samples]

    correct = 0
    total = 0

    for item in samples:
        q_text = item["question"]
        choices = item["choices"]
        gold_idx = item["answer"]

        choice_scores = []
        for choice in choices:
            prompt = f"Question: {q_text}\nAnswer:"
            full_text = f"{prompt} {choice}"
            prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            full_tokens = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

            prefix_len = prompt_tokens.shape[1]
            if full_tokens.shape[1] <= prefix_len:
                choice_scores.append(-1e9)
                continue

            with torch.no_grad():
                logits = model(full_tokens).logits
                target_ids = full_tokens[:, prefix_len:].contiguous()
                target_logits = logits[:, prefix_len - 1 : -1, :].contiguous()
                log_probs = F.log_softmax(target_logits, dim=-1)
                choice_ll = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).sum().item()
                choice_scores.append(choice_ll / max(1, target_ids.shape[1]))

        pred_idx = int(torch.tensor(choice_scores).argmax().item())
        if pred_idx == gold_idx:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    log_step(f"MMLU Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc, total


@torch.no_grad()
def evaluate_boolq(model, tokenizer, num_samples=60, device="cuda"):
    """
    Evaluates reading comprehension yes/no question answering on BoolQ.
    """
    log_step(f"Evaluating BoolQ Reading Comprehension ({num_samples} questions)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("google/boolq", split="validation")
        samples = list(ds)[:num_samples]
        DATASET_PROVENANCE["boolq"] = {
            "source": "Hugging Face google/boolq validation",
            "fallback": False,
            "examples": len(samples),
        }
    except Exception as e:
        log_step(f"datasets BoolQ load failed ({e}), using curated BoolQ sample pool...")
        DATASET_PROVENANCE["boolq"] = {
            "source": "handwritten curated questions",
            "fallback": True,
            "reason": str(e),
        }
        samples = [
            {
                "passage": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
                "question": "does photosynthesis convert light into chemical energy",
                "answer": True,
            },
            {
                "passage": "Mercury is the smallest planet in the Solar System and the closest to the Sun. Its orbit around the Sun takes 87.97 Earth days.",
                "question": "is mercury the farthest planet from the sun",
                "answer": False,
            },
            {
                "passage": "Water freezes at 0 degrees Celsius (32 degrees Fahrenheit) under standard atmospheric pressure.",
                "question": "does water freeze at 0 degrees celsius",
                "answer": True,
            },
        ] * (num_samples // 3 + 1)
        samples = samples[:num_samples]

    correct = 0
    total = 0

    yes_id = tokenizer.encode(" yes", add_special_tokens=False)[-1]
    no_id = tokenizer.encode(" no", add_special_tokens=False)[-1]

    for item in samples:
        passage = item["passage"]
        question = item["question"]
        gold_ans = item["answer"]

        prompt = f"Passage: {passage}\nQuestion: {question}?\nAnswer (yes or no):"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
            prob_yes = logits[0, yes_id].item()
            prob_no = logits[0, no_id].item()

        pred = (prob_yes > prob_no)
        if pred == gold_ans:
            correct += 1
        total += 1

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    log_step(f"BoolQ Accuracy: {acc:.2f}% ({correct}/{total})")
    return acc, total


def evaluate_tier(
    tier_name,
    checkpoint_dir,
    model,
    tokenizer,
    device,
    pt_dtype,
    run_all_tasks=True,
    num_eval_tokens=10000,
):
    """
    Evaluates one tier on all multi-task benchmarks.
    """
    log_step(f"=== Starting Multi-Task Evaluation for Tier '{tier_name}' ===")
    DATASET_PROVENANCE.clear()

    if checkpoint_dir is not None:
        injected = inject_nox_checkpoints(model, checkpoint_dir, device=device, dtype=pt_dtype)
        log_step(f"Injected {len(injected)} compressed layers from '{checkpoint_dir}'")

    # 1. WikiText-2 PPL
    w_text, wikitext_metadata = load_wikitext2_test_text(return_metadata=True)
    DATASET_PROVENANCE["wikitext2"] = {
        "source": wikitext_metadata["source_url"],
        "split": wikitext_metadata["split"],
        "sha256": wikitext_metadata["source_sha256"],
        "fallback": False,
    }
    w_ppl, w_tokens = compute_perplexity(model, tokenizer, w_text, max_eval_tokens=num_eval_tokens, device=device)
    log_step(f"[{tier_name}] WikiText-2 Perplexity: {w_ppl:.2f}")

    # 2. C4 PPL
    c4_text = load_c4_validation_text(max_tokens=num_eval_tokens)
    c4_ppl, c4_tokens = compute_perplexity(model, tokenizer, c4_text, max_eval_tokens=num_eval_tokens, device=device)
    log_step(f"[{tier_name}] C4 Web Perplexity: {c4_ppl:.2f}")

    results = {
        "tier": tier_name,
        "wikitext2_ppl": round(w_ppl, 3),
        "wikitext2_scored_tokens": w_tokens,
        "c4_ppl": round(c4_ppl, 3),
        "c4_scored_tokens": c4_tokens,
    }
    if checkpoint_dir is not None:
        results["serialized_checkpoint_bytes"] = checkpoint_storage_bytes(checkpoint_dir)

    if run_all_tasks:
        # 3. MMLU
        mmlu_acc, mmlu_n = evaluate_mmlu(model, tokenizer, num_samples=60, device=device)
        results["mmlu_acc"] = round(mmlu_acc, 2)

        # 4. GSM8K Math
        gsm_acc, gsm_n = evaluate_gsm8k(model, tokenizer, num_samples=30, device=device)
        results["gsm8k_acc"] = round(gsm_acc, 2)

        # 5. HellaSwag
        hs_acc, hs_n = evaluate_hellaswag(model, tokenizer, num_samples=60, device=device)
        results["hellaswag_acc"] = round(hs_acc, 2)

        # 6. BoolQ
        boolq_acc, boolq_n = evaluate_boolq(model, tokenizer, num_samples=60, device=device)
        results["boolq_acc"] = round(boolq_acc, 2)

        # 7. ARC-Challenge
        arc_acc, arc_n = evaluate_arc_challenge(model, tokenizer, num_samples=60, device=device)
        results["arc_challenge_acc"] = round(arc_acc, 2)

    results["dataset_provenance"] = dict(DATASET_PROVENANCE)

    return results


def main():
    parser = argparse.ArgumentParser(description="Day 5 Multi-Task Evaluation Suite for Nox")
    parser.add_argument("--sweep_dir", type=str, default="./checkpoints/day3_day4_cascade/master_cliff_sweep", help="Directory containing tier subdirectories")
    parser.add_argument("--tier", type=str, default=None, help="Specific tier to evaluate (or None for all 3 tiers)")
    parser.add_argument("--model_id", type=str, default="NousResearch/Llama-2-7b-hf", help="Base Hugging Face model ID")
    parser.add_argument("--eval_dense", action="store_true", help="Also evaluate uncompressed dense FP16 baseline")
    parser.add_argument("--output_file", type=str, default="./checkpoints/day3_day4_cascade/master_cliff_sweep/day5_multitask_results.json", help="Path to save JSON results")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto cuda, else cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pt_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

    log_step(f"Loading base model and tokenizer '{args.model_id}' on {device}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=pt_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    all_results = {}

    # Official published canonical LLaMA-2 7B baseline (Meta / Open LLM Leaderboard / CompactifAI paper)
    if args.eval_dense:
        dense_res = evaluate_tier("dense_fp16", None, model, tokenizer, device, pt_dtype, run_all_tasks=True)
        all_results["dense_fp16"] = dense_res
    else:
        all_results["external_dense_reference"] = {
            "tier": "Published dense reference (not run by this command)",
            "measured_by_this_run": False,
            "protocol_comparable_to_local_results": False,
            "wikitext2_ppl": 5.47,
            "c4_ppl": 7.26,
            "mmlu_acc": 45.3,
            "gsm8k_acc": 14.5,
            "hellaswag_acc": 76.2,
            "boolq_acc": 77.8,
            "arc_challenge_acc": 52.1,
        }

    # Tiers
    tiers = [args.tier] if args.tier else ["production_18x", "high_fidelity_10x", "extreme_77x"]

    for t_name in tiers:
        t_dir = os.path.join(args.sweep_dir, t_name)
        if not os.path.exists(t_dir):
            log_step(f"Warning: Directory '{t_dir}' does not exist; skipping tier '{t_name}'.")
            continue

        # Re-initialize clean model before injecting tier weights
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=pt_dtype,
            low_cpu_mem_usage=True,
        ).to(device)

        t_res = evaluate_tier(t_name, t_dir, model, tokenizer, device, pt_dtype, run_all_tasks=True)
        all_results[t_name] = t_res

    # Save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    log_step(f"All Day 5 Multi-Task Results saved to '{args.output_file}'.")

    # Generate Markdown Summary
    print("\n" + "=" * 140)
    print(" NOX DAY 5 HELD-OUT MULTI-TASK EVALUATION SCORECARD (VS COMPACTIFAI)")
    print("=" * 140)
    print(f"{'Configuration':<24} | {'WikiText-2 PPL':<15} | {'C4 PPL':<10} | {'MMLU (%)':<10} | {'GSM8K (%)':<10} | {'HellaSwag (%)':<14} | {'BoolQ (%)':<10} | {'ARC-C (%)':<10}")
    print("-" * 140)

    for k, v in all_results.items():
        w_ppl = v.get("wikitext2_ppl", "N/A")
        c4_ppl = v.get("c4_ppl", "N/A")
        mmlu = v.get("mmlu_acc", "N/A")
        gsm = v.get("gsm8k_acc", "N/A")
        hs = v.get("hellaswag_acc", "N/A")
        boolq = v.get("boolq_acc", "N/A")
        arc = v.get("arc_challenge_acc", "N/A")
        print(f"{k:<24} | {w_ppl:<15} | {c4_ppl:<10} | {mmlu:<10} | {gsm:<10} | {hs:<14} | {boolq:<10} | {arc:<10}")
    print("=" * 140 + "\n")


if __name__ == "__main__":
    main()
