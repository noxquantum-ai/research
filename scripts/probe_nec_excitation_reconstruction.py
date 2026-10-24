#!/usr/bin/env python3
"""Compare fixed-budget NEC excitation constructors on disjoint local data."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compile_packed_schmidt_model import (
    SafetensorReader, layer_tensor_names, dense_projection_inputs, rms_norm,
)
from nox.block_compiler import scaled_dot_product_attention_2d, ridge_compensated_up_target
from nox.packed_schmidt import reconstruct_packed_schmidt, packed_schmidt_coefficient_count_from_artifact, packed_schmidt_payload_bytes_from_artifact
from nox.sparse_schmidt import covariance_feedback_excitations, excitation_indices, refit_sparse_excitations, reconstruct_excitation_operator
from nox.residual_codec import _pack_codes, encode_groupwise


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--activation-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--projections', nargs='+', default=['q_proj', 'down_proj'])
    parser.add_argument('--methods', nargs='+', default=['pcg_mse', 'pcg_nonzero', 'feedback_fixed', 'feedback_support', 'feedback_nonzero'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    config = json.loads((args.model_dir / 'config.json').read_text())
    checkpoint = torch.load(args.checkpoint_dir / 'compressed_layer_0.pt', weights_only=True)
    weights = {k: v.cuda().float() for k, v in SafetensorReader(args.model_dir).load(layer_tensor_names(0)).items()}
    artifacts = checkpoint['compressed_artifacts']
    decoded = {k: reconstruct_packed_schmidt(v, device='cuda') for k, v in artifacts.items()}
    prefix, heads, sequence, eps = 'model.layers.0.', config['num_attention_heads'], 2048, config['rms_norm_eps']

    def dataset(path):
        hidden = torch.load(path, map_location='cuda', weights_only=True).float()
        teacher = dense_projection_inputs(hidden, weights, layer=0, sequence_length=sequence,
                                          n_heads=heads, n_kv_heads=heads, eps=eps)
        n1 = rms_norm(hidden, weights[prefix + 'input_layernorm.weight'], eps)
        context = scaled_dot_product_attention_2d(*(decoded[p] @ n1 for p in ('q_proj', 'k_proj', 'v_proj')), heads, heads, sequence)
        residual = hidden + decoded['o_proj'] @ context
        n2 = rms_norm(residual, weights[prefix + 'post_attention_layernorm.weight'], eps)
        active = F.silu(decoded['gate_proj'] @ n2) * (decoded['up_proj'] @ n2)
        inputs = {**{p: n1 for p in ('q_proj', 'k_proj', 'v_proj')}, 'o_proj': context,
                  'gate_proj': n2, 'up_proj': n2, 'down_proj': active}
        targets = {p: weights[prefix + ('self_attn.' if p in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp.') + p + '.weight'] @ teacher[p] for p in artifacts}
        targets['down_proj'] += hidden + targets['o_proj'] - residual
        targets['up_proj'], _, _ = ridge_compensated_up_target(
            targets['gate_proj'], targets['up_proj'], decoded['gate_proj'] @ n2,
            down_weight=weights[prefix + 'mlp.down_proj.weight'], ridge_ratio=.05)
        return inputs, targets

    fit_inputs, fit_targets = dataset(args.activation_dir / 'calibration_input_layer_0.pt')
    validation_inputs, validation_targets = dataset(args.activation_dir / 'layer_input_validation_layer_0.pt')
    split = 24576
    report = {'schema': 'nec_excitation_constructor_probe_v1', 'layer': 0,
              'fit_tokens': split, 'guard_tokens': 8192, 'validation_tokens': 8192,
              'selection_data': 'train guard only; validation is reported afterward',
              'baseline_checkpoint_dir': str(args.checkpoint_dir), 'projections': {}}
    selected = copy.deepcopy(checkpoint)
    for role in args.projections:
        began = time.monotonic()
        artifact = artifacts[role]
        x, y = fit_inputs[role], fit_targets[role]
        vx, vy = validation_inputs[role], validation_targets[role]
        base = reconstruct_packed_schmidt(artifact['base_artifact'], device='cuda')
        source = weights[prefix + ('self_attn.' if role in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp.') + role + '.weight']

        def score(candidate):
            w = reconstruct_packed_schmidt(candidate, device='cuda')
            def error(a, b):
                return float((w @ a - b).square().sum() / b.square().sum().clamp_min(1e-30))
            return {'fit_relative_mse': error(x[:, :split], y[:, :split]),
                    'guard_relative_mse': error(x[:, split:], y[:, split:]),
                    'validation_relative_mse': error(vx, vy)}

        baseline = score(artifact)
        best, best_scores, best_name = artifact, baseline, 'baseline'
        trials = []
        prior, delta, indices, solve_report = refit_sparse_excitations(artifact, x, y, split=split)
        for method in args.methods:
            candidates = []
            if method.endswith('_canonical'):
                candidate, _, construction = reconstruct_excitation_operator(
                    source, artifact, x, y, methods=(method,))
                candidates=[(candidate,construction)]
            elif method.startswith('pcg_'):
                grid = 'nonzero' if method == 'pcg_nonzero' else 'symmetric'
                for fraction in (0., .25, .5, 1.):
                    codec = encode_groupwise(prior + fraction * delta,
                        bits=artifact['amplitude_codec']['bits'], group_size=64,
                        scale_policy='mse', quant_grid=grid,
                        entry_weights=x[:, :split].square().mean(dim=1)[indices])
                    candidates.append(({**artifact, 'amplitude_codec': codec}, {'fraction': fraction, **solve_report}))
            else:
                support, codec, construction = covariance_feedback_excitations(
                    source - base, x[:, :split], y[:, :split] - base @ x[:, :split],
                    kept=artifact['amplitudes_per_row'], bits=artifact['amplitude_codec']['bits'],
                    fixed_indices=indices if method == 'feedback_fixed' else None,
                    quant_grid='nonzero' if method == 'feedback_nonzero' else 'symmetric')
                mask = torch.zeros_like(source, dtype=torch.bool).scatter_(1, support, True)
                candidates.append(({**artifact, 'amplitude_codec': codec, 'support_bits': _pack_codes(mask, 1).cpu()}, construction))
            for candidate, construction in candidates:
                if (packed_schmidt_coefficient_count_from_artifact(candidate) != packed_schmidt_coefficient_count_from_artifact(artifact)
                    or packed_schmidt_payload_bytes_from_artifact(candidate) != packed_schmidt_payload_bytes_from_artifact(artifact)):
                    raise RuntimeError('constructor changed its representation budget')
                scores = score(candidate)
                trials.append({'method': method, **construction, **scores})
                if (scores['guard_relative_mse'] < best_scores['guard_relative_mse']
                        and scores['fit_relative_mse'] < baseline['fit_relative_mse']):
                    best, best_scores, best_name = candidate, scores, method
                print(json.dumps({'projection': role, 'method': method, **scores}), flush=True)
            del candidates
        selected['compressed_artifacts'][role] = best
        report['projections'][role] = {'baseline': baseline, 'selected': best_name,
                                     'selected_scores': best_scores, 'trials': trials,
                                     'seconds': time.monotonic() - began}
        save_json(args.output / 'probe.json', report)
        torch.save(selected, args.output / 'compressed_layer_0.pt')
    report['validation_used_for_selection'] = False
    save_json(args.output / 'probe.json', report)


if __name__ == '__main__':
    main()
