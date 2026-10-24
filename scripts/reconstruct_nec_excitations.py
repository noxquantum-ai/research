#!/usr/bin/env python3
"""Reconstruct selected NEC amplitudes from dense training trajectories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compile_packed_schmidt_model import SafetensorReader, rms_norm, save_checkpoint_atomically
from nox.sparse_schmidt import reconstruct_excitation_operator


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--activation-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--projections', nargs='+', default=['q_proj', 'k_proj'])
    p.add_argument('--methods', nargs='+', default=['pcg_mse', 'pcg_nonzero', 'feedback_fixed', 'feedback_support', 'feedback_nonzero'])
    p.add_argument('--layers', type=int, nargs='+', default=list(range(32)))
    args = p.parse_args()
    if set(args.projections) - {'q_proj', 'k_proj', 'v_proj'}:
        p.error('this independent control supports Q/K/V, which share the captured dense input')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    reader = SafetensorReader(args.model_dir)
    config = json.loads((args.model_dir / 'config.json').read_text())
    report = {'schema': 'nec_covariance_excitation_reconstruction_v1', 'scope': 'full original model',
              'source_checkpoints': str(args.checkpoint_dir), 'construction_trajectory': 'dense',
              'projections': args.projections, 'layers': [], 'uses_autograd': False,
              'logical_coefficients_added': 0, 'tensor_payload_bytes_added': 0}
    for layer in range(32):
        path = args.output / f'compressed_layer_{layer}.pt'
        if layer not in args.layers:
            if not path.exists(): path.symlink_to((args.checkpoint_dir / path.name).resolve())
            continue
        if path.exists():
            raise FileExistsError(path)
        began = time.monotonic()
        checkpoint = torch.load(args.checkpoint_dir / path.name, weights_only=True)
        prefix = f'model.layers.{layer}.'
        names = [prefix+'input_layernorm.weight'] + [prefix+'self_attn.'+r+'.weight' for r in args.projections]
        w = {k:v.cuda().float() for k,v in reader.load(names).items()}
        hidden = torch.load(args.activation_dir / f'calibration_input_layer_{layer}.pt', map_location='cuda', weights_only=True).float()
        x = rms_norm(hidden, w[prefix+'input_layernorm.weight'], config['rms_norm_eps'])
        row = {'layer':layer,'projections':{}}
        for role in args.projections:
            weight = w[prefix+'self_attn.'+role+'.weight']
            artifact, decoded, diagnostics = reconstruct_excitation_operator(weight,
                checkpoint['compressed_artifacts'][role], x, weight @ x, methods=args.methods)
            checkpoint['compressed_artifacts'][role] = artifact
            row['projections'][role] = diagnostics
            print(json.dumps({'layer':layer,'projection':role,'selected':diagnostics['selected'],
                              'guard_before':diagnostics['initial_guard_relative_mse'],
                              'guard_after':diagnostics['final_guard_relative_mse']}), flush=True)
            del artifact, decoded
        save_checkpoint_atomically(checkpoint, path)
        row['seconds'] = time.monotonic()-began
        report['layers'].append(row)
        temp = args.output/'construction.json.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output/'construction.json')
        del checkpoint, w, x, hidden


if __name__ == '__main__': main()
