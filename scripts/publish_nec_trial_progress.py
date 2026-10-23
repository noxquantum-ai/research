#!/usr/bin/env python3
"""Archive measured NEC stages from a pod using an existing SSH connection.

Only JSON reports are copied. A detached local Git worktree keeps publication
separate from source edits, and pushes are ordinary fast-forward updates.
Credentials stay in the SSH control connection and existing Git configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
REMOTE_READ = r'''
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); result={"files":{}}
for name in ["status.json","run_environment.json","activation_capture_manifest.json","model_manifest.json"]:
 q=p/name
 if q.is_file():
  try: result["files"][name]=json.loads(q.read_text())
  except (ValueError,OSError): pass
for q in p.glob("evaluation_stage*.json"):
 try: result["files"][q.name]=json.loads(q.read_text())
 except (ValueError,OSError): pass
for name in ["package_measurement.json","parity.json"]:
 q=p/"export"/name
 if q.is_file():
  try: result["files"][name]=json.loads(q.read_text())
  except (ValueError,OSError): pass
print(json.dumps(result,separators=(",",":")))
'''


def run(argv, *, cwd=ROOT):
    return subprocess.run(argv, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=120).stdout


def report_readme(status):
    lines = ['# Live hybrid NEC trial reports', '',
             f"Snapshot phase: `{status.get('phase', 'unknown')}`; UTC: {status.get('updated_utc', 'unknown')}.", '',
             'These are measured checkpoints from the new pod. Partial checkpoints retain',
             'all uncompiled layers dense and do not demonstrate whole-model compression.', '',
             '| Compressed layers | Scored tokens | Dense PPL | NEC PPL | KL (nats/token) | Next-token accuracy | Teacher top-1 agreement |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in status.get('evaluations', []):
        s, t = row['student'], row['teacher']
        lines.append(f"| 0–{row['compressed_layer_count'] - 1} | {row['scored_tokens']} | {t['perplexity']:.5f} | {s['perplexity']:.5f} | {s['teacher_kl_nats_per_token']:.5f} | {100*s['next_token_accuracy']:.2f}% | {100*s['dense_top1_agreement']:.2f}% |")
    if status.get('final_heldout_evaluation'):
        q = status['final_heldout_evaluation']
        lines.extend(['', f"Fresh held-out test range: {q['scored_tokens']:,} scored tokens; "
                      f"dense PPL {q['teacher']['perplexity']:.5f}; "
                      f"NEC PPL {q['student']['perplexity']:.5f}; "
                      f"KL {q['student']['teacher_kl_nats_per_token']:.5f} nats/token."])
    if status.get('actual_whole_model'):
        m = status['actual_whole_model']
        lines.extend(['', f"Actual complete package: {m['serialized_bytes']:,} bytes; "
                      f"{m['compression_ratio']:.5f}× storage compression; "
                      f"{m['structural_reduction']:.5f}× structural reduction."])
    lines.extend(['', '**All targets demonstrated together: ' + ('YES' if status.get('success_demonstrated') else 'NOT YET') + '.**', '',
                  'The gate requires all 32 layers, independent held-out PPL ≤10, actual',
                  'complete-package compression ≥5×, structural reduction ≥2×, and exported',
                  'runtime parity. No gradient-based healing is used. Runtime reconstructs',
                  'dense matrices, so storage reduction does not imply runtime memory savings.'])
    if status.get('error'):
        lines.extend(['', 'The run stopped with an error; see [status.json](status.json).'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--control-socket', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=40)
    args = parser.parse_args()
    if not args.worktree.exists():
        run(['git', 'fetch', 'origin', 'main'])
        run(['git', 'worktree', 'add', '--detach', str(args.worktree), 'origin/main'])
    destination = args.worktree / 'results/nec_sparse_excitation_a40_20261023/live'
    destination.mkdir(parents=True, exist_ok=True)
    last_trigger = None
    while True:
        try:
            remote_command = 'python -c ' + shlex.quote(REMOTE_READ) + ' ' + shlex.quote(args.run_dir)
            data = json.loads(run(['ssh', '-S', args.control_socket, '-o', 'BatchMode=yes',
                                   '-p', str(args.port), args.host, remote_command]))['files']
            status = data.get('status.json')
            if status is None:
                time.sleep(args.interval)
                continue
            # Preserve complete stage manifests before the next compile replaces them.
            manifest = data.pop('model_manifest.json', None)
            if manifest is not None:
                n = len(manifest['compiled_layers'])
                data[f'compile_manifest_stage{n}.json'] = manifest
            for name, value in data.items():
                (destination / name).write_text(json.dumps(value, indent=2) + '\n')
            terminal = status.get('phase') in ('finished', 'failed')
            trigger = (len(status.get('evaluations', [])), status['phase'] if terminal else None)
            if trigger != last_trigger:
                (destination / 'README.md').write_text(report_readme(status))
                run(['git', 'add', 'results/nec_sparse_excitation_a40_20261023/live'], cwd=args.worktree)
                changed = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=args.worktree).returncode
                if changed:
                    count = status.get('evaluations', [{}])[-1].get('compressed_layer_count', 0) if status.get('evaluations') else 0
                    run(['git', 'commit', '-m', f'Record NEC hybrid trial progress through {count} evaluated layers'], cwd=args.worktree)
                    for attempt in range(3):
                        try:
                            run(['git', 'fetch', 'origin', 'main'], cwd=args.worktree)
                            run(['git', 'rebase', 'origin/main'], cwd=args.worktree)
                            run(['git', 'push', 'origin', 'HEAD:main'], cwd=args.worktree)
                            break
                        except subprocess.CalledProcessError:
                            if attempt == 2:
                                raise
                    print(f"Published phase={status['phase']} evaluated_layers={count}", flush=True)
                last_trigger = trigger
            if terminal:
                return
        except (subprocess.SubprocessError, OSError, ValueError, KeyError) as error:
            print(f'Publication retry: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
