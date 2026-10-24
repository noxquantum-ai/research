#!/usr/bin/env python3
"""Test function-preserving SwiGLU ordering at existing MPO Schmidt ranks.

This is a representation screen, not a deployable compressed model or a PPL
result. Training-only neuron contribution correlations define one shared
ordering for Gate/Up rows and Down columns. A randomized ordering is a control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nox.operator_schmidt import _layout_from_value, _matrix_to_unfolding, _truncated_svd
from nox.residual_codec import decode_groupwise
from scripts.compile_packed_schmidt_model import SafetensorReader, dense_projection_inputs, layer_tensor_names


def contribution_gram(active, down):
    """Exact Gram of neuron functions phi_i(x)=Down[:,i]*active[i,x]."""
    return (active @ active.T / active.shape[1]) * (down.T @ down)


def spectral_order(gram, *, seed=271828):
    adjacency = gram.abs().clone()
    adjacency.diagonal().zero_()
    degree = adjacency.sum(dim=1).clamp_min(1e-20)
    scale = degree.rsqrt()
    adjacency.mul_(scale[:, None]).mul_(scale[None, :])
    generator = torch.Generator(device=gram.device).manual_seed(seed)
    initial = torch.randn(len(gram), 3, generator=generator, device=gram.device, dtype=gram.dtype)
    values, vectors = torch.lobpcg(adjacency, k=3, X=initial, niter=80, tol=1e-5, largest=True)
    # Fix the eigenvector sign so the recipe is reproducible.
    axis = vectors[:, 1] * scale
    pivot = axis.abs().argmax()
    if axis[pivot] < 0:
        axis = -axis
    return axis.argsort(), values.tolist()


def permute_swiglu(weights, order):
    width = weights['gate_proj'].shape[0]
    if order.ndim != 1 or not torch.equal(order.sort().values, torch.arange(width, device=order.device)):
        raise ValueError('neuron order must be a complete bijection')
    return {
        'gate_proj': weights['gate_proj'][order],
        'up_proj': weights['up_proj'][order],
        'down_proj': weights['down_proj'][:, order],
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--activation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--fit-tokens', type=int, default=8192)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    start = time.monotonic()
    config = json.loads((args.model_dir / 'config.json').read_text())
    capture = json.loads((args.activation_dir / 'model_manifest.json').read_text())
    sequence = int(capture['sequence_length'])
    if args.fit_tokens % sequence or args.fit_tokens > 24576:
        raise ValueError('use complete construction-fit windows only')
    source = SafetensorReader(args.model_dir).load(layer_tensor_names(args.layer))
    source = {k: v.float() for k, v in source.items()}
    path = args.activation_dir / f'calibration_input_layer_{args.layer}.pt'
    hidden = torch.load(path, map_location='cpu', weights_only=True)[:, :args.fit_tokens].float()
    inputs = dense_projection_inputs(hidden, source, layer=args.layer, sequence_length=sequence,
                                     n_heads=config['num_attention_heads'],
                                     n_kv_heads=config.get('num_key_value_heads', config['num_attention_heads']),
                                     eps=config['rms_norm_eps'])
    prefix = f'model.layers.{args.layer}.mlp.'
    weights = {p: source[prefix + p + '.weight'] for p in ('gate_proj', 'up_proj', 'down_proj')}
    gram = contribution_gram(inputs['down_proj'], weights['down_proj'])
    order, eigenvalues = spectral_order(gram)
    del gram
    generator = torch.Generator().manual_seed(314159)
    orders = {'identity': torch.arange(len(order)), 'functional_spectral': order,
              'random_control': torch.randperm(len(order), generator=generator)}
    checkpoint = torch.load(args.checkpoint_dir / f'compressed_layer_{args.layer}.pt',
                            map_location='cpu', weights_only=True)
    report = {
        'schema': 'nec_swiglu_ordering_screen_v1', 'layer': args.layer,
        'fit_tokens': args.fit_tokens, 'data_split': 'construction train fit only',
        'validation_used_for_selection': False, 'uses_autograd': False,
        'deployable_model_produced': False, 'end_to_end_quality_not_measured': True,
        'graph_eigenvalues': eigenvalues, 'orders': {},
        'prior_art': 'https://arxiv.org/abs/2401.16367; no distillation or training used here',
    }
    for name, permutation in orders.items():
        changed = permute_swiglu(weights, permutation)
        x = inputs['gate_proj'][:, :32]
        evaluate = lambda w: w['down_proj'] @ (F.silu(w['gate_proj'] @ x) * (w['up_proj'] @ x))
        reference, actual = evaluate(weights), evaluate(changed)
        record = {
            'permutation_sha256': hashlib.sha256(permutation.numpy().tobytes()).hexdigest(),
            'int16_permutation_bytes_if_stored_once': len(order) * 2,
            'dense_function_relative_error': float((actual-reference).norm()/reference.norm()),
            'projections': {},
        }
        for role, matrix in changed.items():
            base = checkpoint['compressed_artifacts'][role]['base_artifact']
            layout = _layout_from_value(base['layout'])
            rank = decode_groupwise(base['left_codec']).shape[0]
            scale = base['input_scale'].float()
            if role == 'down_proj':
                scale = scale[permutation]
            unfolding = _matrix_to_unfolding(matrix * scale[None, :], layout)
            left, singular, right = _truncated_svd(unfolding, rank, oversample=32,
                                                  power_iterations=4, seed=161803)
            record['projections'][role] = {
                'same_schmidt_rank': rank,
                'weighted_frobenius_relative_squared_error': float(
                    (unfolding - (left * singular) @ right).square().sum() / unfolding.square().sum()
                ),
            }
            del unfolding, left, singular, right
        report['orders'][name] = record
        report['seconds'] = time.monotonic()-start
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'order': name, **record}), flush=True)


if __name__ == '__main__':
    main()
