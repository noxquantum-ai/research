#!/usr/bin/env python3
"""Continue an interrupted/staged NEC trial without an interactive session.

Uses the captured original compiler argv, replays saved compressed prefixes,
records status atomically, and never changes the requested model budgets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def active_trial_processes(run_dir):
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().split(b'\0')
            values = [x.decode(errors='replace') for x in argv if x]
            if (str(run_dir) in values and any(
                Path(x).name in ('compile_packed_schmidt_model.py', 'evaluate_packed_teacher_kl.py')
                for x in values
            )):
                result.append(int(entry.name))
        except (OSError, ValueError):
            pass
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--compiler-argv', type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    argv = json.loads(args.compiler_argv.read_text())
    model_dir = argv[argv.index('--model-dir') + 1]
    python = argv[0]
    activation_dir = Path(argv[argv.index('--activation-dir') + 1])
    shutil.copy2(activation_dir / 'model_manifest.json', run_dir / 'activation_capture_manifest.json')
    atomic_json(run_dir / 'run_environment.json', {
        'compiler_argv': argv,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint_directory_is_persistent': str(run_dir).startswith('/workspace/'),
    })
    state = {'schema': 'nec_unattended_trial_status_v1', 'run_dir': str(run_dir),
             'target': {'layers': 32, 'maximum_ppl': 10, 'minimum_storage_ratio': 5,
                        'minimum_structural_reduction': 2},
             'evaluations': [], 'success_demonstrated': False}

    def update(phase, **extra):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        state['completed_layers'] = sorted(int(p.stem.removeprefix('compressed_layer_'))
                                           for p in run_dir.glob('compressed_layer_*.pt'))
        atomic_json(run_dir / 'status.json', state)

    def execute(command, log, phase):
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                update(phase, child_pid=process.pid, current_log=str(log))
                time.sleep(10)
            if process.returncode:
                raise RuntimeError(f'{phase} exited {process.returncode}; see {log}')

    try:
        # Adopt the existing stage instead of competing with its GPU process.
        while active_trial_processes(run_dir):
            update('waiting_for_existing_stage')
            time.sleep(10)
        for count in (8, 16, 24, 32):
            result_path = run_dir / f'evaluation_stage{count}_validation_4k.json'
            if not result_path.is_file():
                stage_argv = list(argv)
                stage_argv[stage_argv.index('--layers') + 1] = f'0-{count - 1}'
                if '--resume' not in stage_argv:
                    stage_argv.append('--resume')
                execute(stage_argv, run_dir / f'compile_stage{count}.log', f'compiling_stage{count}')
                execute([python, str(ROOT / 'scripts/evaluate_packed_teacher_kl.py'),
                         '--checkpoint-dir', str(run_dir), '--model-id', model_dir,
                         '--split', 'validation', '--max-eval-tokens', '4096',
                         '--output', str(result_path)],
                        run_dir / f'eval_stage{count}.log', f'evaluating_stage{count}')
            result = json.loads(result_path.read_text())
            if result['compressed_layers'] != list(range(count)):
                raise ValueError(f'stage {count} evaluation uses a different layer set')
            state['evaluations'].append({'compressed_layer_count': count, 'file': str(result_path),
                                         'scored_tokens': result['scored_tokens'],
                                         'teacher': result['teacher'], 'student': result['student']})
            update(f'stage{count}_measured')

        quality_path = run_dir / 'evaluation_stage32_validation_4k.json'
        if state['evaluations'][-1]['student']['perplexity'] <= 10:
            # Fresh held-out range, beyond historical short test diagnostics.
            quality_path = run_dir / 'evaluation_stage32_heldout_test.json'
            execute([python, str(ROOT / 'scripts/evaluate_packed_teacher_kl.py'),
                     '--checkpoint-dir', str(run_dir), '--model-id', model_dir,
                     '--split', 'test', '--token-offset', '32768',
                     '--max-eval-tokens', '1000000000', '--output', str(quality_path)],
                    run_dir / 'eval_heldout_test.log', 'evaluating_fresh_heldout_test')
            state['final_heldout_evaluation'] = json.loads(quality_path.read_text())
        execute([python, str(ROOT / 'scripts/export_packed_schmidt_model.py'),
                 '--model-dir', model_dir, '--checkpoint-dir', str(run_dir),
                 '--quality-report', str(quality_path), '--output', str(run_dir / 'export')],
                run_dir / 'export.log', 'measuring_actual_complete_package')
        measurement = json.loads((run_dir / 'export/package_measurement.json').read_text())
        state['actual_whole_model'] = {k: measurement[k] for k in
                                      ('serialized_bytes', 'compression_ratio', 'structural_reduction', 'acceptance')}
        execute([python, str(ROOT / 'scripts/verify_packed_nec_package.py'),
                 '--model-dir', model_dir, '--checkpoint-dir', str(run_dir),
                 '--package-dir', str(run_dir / 'export/deployable_model'),
                 '--output', str(run_dir / 'export/parity.json')],
                run_dir / 'export_parity.log', 'checking_exported_runtime_parity')
        state['export_parity'] = json.loads((run_dir / 'export/parity.json').read_text())
        # Never turn a development slice into final confirmation.
        state['success_demonstrated'] = (
            'final_heldout_evaluation' in state
            and measurement['acceptance']['status'] == 'FEASIBLE'
            and state['export_parity']['passed']
        )
        update('finished', child_pid=None)
    except BaseException:
        update('failed', error=traceback.format_exc(), child_pid=None)
        raise


if __name__ == '__main__':
    main()
