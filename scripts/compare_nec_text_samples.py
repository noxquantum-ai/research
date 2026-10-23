#!/usr/bin/env python3
"""Save paired greedy examples; these are qualitative checks, not a benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    questions = ['What is the capital of France?', 'What is 17 multiplied by 23?',
                 'Which planet is called the Red Planet?',
                 'At what temperature in degrees Celsius does water freeze at standard pressure?',
                 'What is the sum of the integers from 1 through 5?']
    prompts = ['Q: What is 2 plus 2?\nA: 4.\n\nQ: What is the capital of Italy?\nA: Rome.\n\nQ: ' + q + '\nA:'
               for q in questions]
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float16, device_map='auto', local_files_only=True,
    ).eval()
    inputs = tokenizer(prompts, padding=True, return_tensors='pt').to(model.device)

    def generate():
        tokens = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                num_beams=1, use_cache=True, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

    teacher = generate()
    layers = inject_nox_checkpoints(model, args.checkpoint_dir, device='cuda', dtype=torch.float16)
    student = generate()
    report = {
        'schema': 'nec_paired_greedy_examples_v1', 'compressed_layers': layers,
        'checkpoint_dir': str(args.checkpoint_dir.resolve()),
        'semantics': 'Qualitative development prompts; no public benchmark or accuracy score is implied.',
        'decoding': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': 64},
        'examples': [{'prompt': p, 'teacher_raw_completion': t, 'student_raw_completion': s}
                     for p, t, s in zip(prompts, teacher, student)],
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
