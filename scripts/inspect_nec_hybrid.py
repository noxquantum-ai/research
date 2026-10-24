#!/usr/bin/env python3
"""Count hybrid NEC components and prepare a causal MPO-core ablation.

The ablation zeroes only main MPO core group scales. Excitation amplitudes,
supports, other branches and every tensor shape stay unchanged. This measures
contribution within a constructed artifact; it is not a reoptimized sparse
baseline and must not be described as one.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nox.packed_schmidt import (
    packed_codec_coefficient_count, packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact,
)
from nox.residual_codec import codec_payload_bytes


def component_counts(artifact):
    if artifact.get('codec_type') not in ('nec_sparse_excitation_schmidt_v1','nec_sparse_excitation_schmidt_v2'):
        raise ValueError('component breakdown requires the sparse-excitation hybrid')
    base, amplitudes = artifact['base_artifact'], artifact['amplitude_codec']
    sparse_values = int(amplitudes['shape'][0]) * int(amplitudes['shape'][1])
    main_core_values = sum(int(base[name]['shape'][0]) * int(base[name]['shape'][1])
                          for name in ('left_codec', 'right_codec'))
    core_auxiliaries = sum(packed_codec_coefficient_count(base[name])
                           for name in ('left_codec', 'right_codec')) - main_core_values
    core_auxiliaries += base['input_scale'].numel()
    if base.get('output_scale') is not None:
        core_auxiliaries += base['output_scale'].numel()
    total = packed_schmidt_coefficient_count_from_artifact(artifact)
    base_total = packed_schmidt_coefficient_count_from_artifact(base)
    row = {
        'main_mpo_core_values': main_core_values,
        'main_mpo_scale_coefficients': core_auxiliaries,
        'other_backbone_branch_coefficients': base_total - main_core_values - core_auxiliaries,
        'sparse_amplitude_values': sparse_values,
        'sparse_scale_coefficients': packed_codec_coefficient_count(amplitudes) - sparse_values,
        'total_artifact_coefficients': total,
        'support_mask_bytes': artifact['support_bits'].numel() * artifact['support_bits'].element_size(),
        'backbone_tensor_payload_bytes': packed_schmidt_payload_bytes_from_artifact(base),
        'sparse_value_and_scale_payload_bytes': codec_payload_bytes(amplitudes),
        'total_tensor_payload_bytes': packed_schmidt_payload_bytes_from_artifact(artifact),
    }
    coefficient_parts = ('main_mpo_core_values', 'main_mpo_scale_coefficients',
                         'other_backbone_branch_coefficients', 'sparse_amplitude_values', 'sparse_scale_coefficients')
    if sum(row[k] for k in coefficient_parts) != total:
        raise RuntimeError('component coefficient partition does not match the artifact count')
    if (row['support_mask_bytes'] + row['backbone_tensor_payload_bytes']
            + row['sparse_value_and_scale_payload_bytes'] != row['total_tensor_payload_bytes']):
        raise RuntimeError('component byte partition does not match the artifact payload')
    return row


def zero_main_mpo_cores(artifact):
    result = copy.deepcopy(artifact)
    if result.get('codec_type') not in ('nec_sparse_excitation_schmidt_v1','nec_sparse_excitation_schmidt_v2'):
        raise ValueError('MPO-core ablation requires a sparse-excitation hybrid')
    for name in ('left_codec', 'right_codec'):
        result['base_artifact'][name]['group_scales'].zero_()
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zero-main-mpo-output', type=Path)
    args = parser.parse_args()
    if args.zero_main_mpo_output is not None:
        args.zero_main_mpo_output.mkdir(parents=True, exist_ok=False)
    rows, totals = [], {}
    for path in sorted(args.checkpoint_dir.glob('compressed_layer_*.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        layer = {'layer': int(checkpoint['layer_idx']), 'projections': {}}
        for name, artifact in checkpoint['compressed_artifacts'].items():
            counts = component_counts(artifact)
            layer['projections'][name] = counts
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
            if args.zero_main_mpo_output is not None:
                checkpoint['compressed_artifacts'][name] = zero_main_mpo_cores(artifact)
        rows.append(layer)
        if args.zero_main_mpo_output is not None:
            checkpoint['experimental_ablation'] = 'main_mpo_cores_zeroed_without_reallocating_budget'
            torch.save(checkpoint, args.zero_main_mpo_output / path.name)
    if not rows:
        raise ValueError('no compressed checkpoints found')
    report = {'schema': 'nec_hybrid_component_breakdown_v1', 'layers': rows, 'totals': totals,
              'scope': 'artifact components only; retained source tensors belong in complete-package accounting',
              'causal_ablation_directory': str(args.zero_main_mpo_output) if args.zero_main_mpo_output else None,
              'ablation_is_reoptimized_matched_sparse_baseline': False}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    main()
