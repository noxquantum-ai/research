#!/usr/bin/env python3
"""Queue guarded NEC construction selection, matched evaluation and export.

Runs only after a specified construction status reaches its terminal phase.
Training guards select constructions; validation and full test regression
measure them independently. This never publishes a model or marks an unmet
dense-quality target as successful.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def compare(reference, candidate):
    if reference['input_token_ids_sha256'] != candidate['input_token_ids_sha256']:
        raise ValueError('quality comparisons require identical token IDs')
    if reference['teacher'] != candidate['teacher']:
        raise ValueError('matched dense teachers differ')
    a, b = reference['student'], candidate['student']
    return {
        **{k: b[k] <= a[k] for k in ('perplexity', 'teacher_kl_nats_per_token')},
        **{k: b[k] >= a[k] for k in ('next_token_accuracy', 'dense_top1_agreement')},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'baseline-dir', 'alternative-dir', 'capture-manifest',
                 'baseline-validation', 'baseline-test', 'report-dir', 'assembly-dir',
                 'export-dir', 'wait-status'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--wait-phase', default='validation_measured')
    parser.add_argument('--maximum-bytes', type=int, default=2490273288)
    parser.add_argument('--maximum-coefficients', type=int, default=3323032892)
    args = parser.parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.report_dir / 'status.json'
    if status_path.exists():
        raise FileExistsError('this selection run already has a status file')
    state = {
        'schema': 'nec_guarded_accuracy_selection_v1',
        'final_success_demonstrated': False,
        'maximum_complete_serialized_bytes': args.maximum_bytes,
        'maximum_complete_coefficients': args.maximum_coefficients,
        'construction_status': str(args.wait_status),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }

    def update(phase, **extra):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(status_path, state)

    def execute(script, argv, name, phase):
        command = [sys.executable, str(ROOT / 'scripts' / script), *map(str, argv)]
        with (args.report_dir / name).open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            update(phase, command=command, child_pid=process.pid)
            if process.wait():
                raise RuntimeError(f'{phase} failed: see {name}')
        update(phase, child_pid=None)

    def evaluate(reference_path, name):
        reference = json.loads(reference_path.read_text())
        destination = args.report_dir / name
        execute('evaluate_packed_teacher_kl.py', [
            '--model-id', args.model_dir, '--checkpoint-dir', args.assembly_dir,
            '--split', reference['test_split']['split'],
            '--token-offset', reference['token_offset'],
            '--max-eval-tokens', reference['input_tokens'],
            '--seq-len', reference['context'], '--stride', reference['stride'],
            '--output', destination,
        ], name + '.log', 'evaluating_' + reference['test_split']['split'])
        result = json.loads(destination.read_text())
        gates = compare(reference, result)
        update('measured_' + name.removesuffix('.json'), quality=result['student'],
               comparison_reference=str(reference_path), non_degradation=gates)
        return result, gates

    try:
        update('waiting_for_construction')
        while True:
            status = json.loads(args.wait_status.read_text())
            if status['phase'] == args.wait_phase:
                # The worker writes its terminal phase after its child exits.
                break
            if status['phase'] == 'failed':
                raise RuntimeError('upstream construction failed')
            time.sleep(15)
        execute('select_nec_operator_constructions.py', [
            '--model-dir', args.model_dir, '--baseline-dir', args.baseline_dir,
            '--alternative-dir', args.alternative_dir,
            '--capture-manifest', args.capture_manifest,
            '--report-dir', args.report_dir, '--assembly-dir', args.assembly_dir,
            '--initial-construction', 'best_guard',
        ], 'selection.log', 'selecting_on_training_guard')
        validation, gates = evaluate(args.baseline_validation, 'validation_8k.json')
        if not all(gates.values()):
            update('rejected_validation_regression')
            return
        final, gates = evaluate(args.baseline_test, 'heldout_test.json')
        if not all(gates.values()):
            update('rejected_test_regression')
            return
        maximum_ppl = final['teacher']['perplexity'] + .5
        execute('export_packed_schmidt_model.py', [
            '--model-dir', args.model_dir, '--checkpoint-dir', args.assembly_dir,
            '--output', args.export_dir, '--quality-report', args.report_dir / 'heldout_test.json',
            '--checkpoint-container', 'gzip', '--retained-container', 'gzip',
            '--max-ppl', maximum_ppl,
        ], 'export.log', 'measuring_complete_package')
        measurement = json.loads((args.export_dir / 'package_measurement.json').read_text())
        package = Path(measurement['path'])
        execute('verify_packed_nec_package.py', [
            '--model-dir', args.model_dir, '--checkpoint-dir', args.assembly_dir,
            '--package-dir', package, '--output', args.report_dir / 'runtime_parity.json',
        ], 'runtime_parity.log', 'verifying_fresh_exported_runtime')
        t, s = final['teacher'], final['student']
        target = {
            'storage': measurement['serialized_bytes'] <= args.maximum_bytes,
            'structure': measurement['complete_model_coefficient_count'] <= args.maximum_coefficients,
            'fallback_ppl': s['perplexity'] <= maximum_ppl,
            'accuracy': abs(s['next_token_accuracy'] - t['next_token_accuracy']) <= .02,
            'agreement': s['dense_top1_agreement'] >= .98,
            'runtime_parity': json.loads((args.report_dir / 'runtime_parity.json').read_text())['passed'],
        }
        measurement['acceptance'] = {
            'maximum_complete_serialized_bytes': args.maximum_bytes,
            'maximum_complete_coefficients': args.maximum_coefficients,
            'maximum_ppl_excess_over_dense': .5,
            'maximum_accuracy_gap_points': 2,
            'minimum_dense_top1_agreement': .98,
            'gates': target, 'all_numerical_targets_met': all(target.values()),
            'persistent_copy_required_before_promotion': True,
        }
        atomic_json(args.report_dir / 'package_measurement.json', measurement)
        atomic_json(args.export_dir / 'package_measurement.json', measurement)
        update('measured', final_gates=target, ideal_dense_ppl_met=s['perplexity'] <= t['perplexity'],
               package_in_scratch=str(package), all_numerical_targets_met=all(target.values()),
               requires_persistent_package_copy_before_promotion=True)
    except BaseException:
        update('failed', child_pid=None, error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
