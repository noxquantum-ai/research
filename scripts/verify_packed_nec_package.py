#!/usr/bin/env python3
"""Verify exported NEC inference against the measured checkpoint assembly."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--package-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    config = json.loads((args.model_dir / 'config.json').read_text())
    generator = torch.Generator().manual_seed(271828)
    ids = torch.randint(0, config['vocab_size'], (1, 128), generator=generator).to(args.device)
    device_map = 'auto' if args.device == 'cuda' else {'': 'cpu'}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float16, device_map=device_map, local_files_only=True,
    ).eval()
    injected = inject_nox_checkpoints(model, args.checkpoint_dir, device=args.device, dtype=torch.float16)
    reference = model(ids, use_cache=False).logits.cpu()
    del model
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    # A fresh interpreter loads the bundled decoder, not the repository's
    # already-imported nox modules. Runtime caches must not modify the package.
    with tempfile.TemporaryDirectory(prefix='nec-export-parity-') as temporary:
        inputs_path, logits_path = Path(temporary) / 'ids.pt', Path(temporary) / 'logits.pt'
        torch.save(ids.cpu(), inputs_path)
        source = (
            'import sys,torch\nfrom load_nec_package import load_model\n'
            'device=sys.argv[3]\n'
            'model=load_model(device_map="auto" if device=="cuda" else {"":"cpu"},torch_dtype=torch.float16)\n'
            'ids=torch.load(sys.argv[1],weights_only=True).to(device)\n'
            'with torch.inference_mode():\n'
            ' torch.save(model(ids,use_cache=False).logits.cpu(),sys.argv[2])\n'
        )
        environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
        environment.pop('PYTHONPATH', None)
        subprocess.run([sys.executable, '-c', source, str(inputs_path), str(logits_path), args.device],
                       cwd=args.package_dir, env=environment, check=True)
        actual = torch.load(logits_path, weights_only=True)
    difference = (reference.float() - actual.float()).abs()
    report = {
        'schema': 'nec_export_runtime_parity_v1', 'checkpoint_dir': str(args.checkpoint_dir.resolve()),
        'package_dir': str(args.package_dir.resolve()), 'injected_layers': injected,
        'test_tokens': ids.numel(), 'bitwise_equal': torch.equal(reference, actual),
        'maximum_absolute_logit_difference': float(difference.max()),
        'top1_agreement': float((reference.argmax(-1) == actual.argmax(-1)).float().mean()),
        'bundled_decoder_tested_in_fresh_interpreter': True,
        'device': args.device,
    }
    report['passed'] = report['bitwise_equal']
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['passed']:
        raise RuntimeError('exported model differs from the evaluated checkpoint assembly')


if __name__ == '__main__':
    main()
