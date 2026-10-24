#!/usr/bin/env python3
"""Run a fixed-budget sequential NEC reconstruction and matched quality gates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--baseline-dir', type=Path, required=True)
    p.add_argument('--compiler-argv', type=Path, required=True)
    args = p.parse_args()
    run, source = args.run_dir.resolve(), args.baseline_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    candidate = run / 'coupled-full32'
    argv = json.loads(args.compiler_argv.read_text())
    python = argv[0]
    model = argv[argv.index('--model-dir') + 1]
    argv[argv.index('--output') + 1] = str(candidate)
    argv.extend(['--reconstruct-from', str(source), '--resume'])
    state = {'schema': 'nec_sequential_accuracy_reconstruction_v1',
             'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'baseline_dir': str(source), 'candidate_dir': str(candidate),
             'maximum_complete_serialized_bytes': 2490273288,
             'maximum_complete_logical_coefficients': 3323032892,
             'target_ppl_increase': 0, 'fallback_maximum_ppl_increase': .5,
             'maximum_accuracy_gap_points': 2, 'minimum_teacher_agreement': .98,
             'final_success_demonstrated': False}

    def update(phase, **extra):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = run/'coupled_status.tmp'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(run/'coupled_status.json')

    def execute(command, name, phase):
        update(phase, command=command)
        with (run/name).open('a') as log:
            child = subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            update(phase, child_pid=child.pid)
            if child.wait(): raise RuntimeError(f'{phase} failed; see {name}')

    def evaluate(directory, result, phase, *, split='validation', offset=0, count=8192):
        if result.exists(): return json.loads(result.read_text())
        execute([python,str(ROOT/'scripts/evaluate_packed_teacher_kl.py'),
                 '--checkpoint-dir',str(directory),'--model-id',model,
                 '--split',split,'--token-offset',str(offset),'--max-eval-tokens',str(count),
                 '--output',str(result)],result.stem+'.log',phase)
        return json.loads(result.read_text())

    def compare(a,b):
        if a['input_token_ids_sha256']!=b['input_token_ids_sha256']:
            raise RuntimeError('baseline/candidate evaluation IDs differ')
        x,y=a['student'],b['student']
        return {**{k:y[k]<=x[k] for k in ('perplexity','teacher_kl_nats_per_token')},
                **{k:y[k]>=x[k] for k in ('next_token_accuracy','dense_top1_agreement')}}

    try:
        baseline = evaluate(source,run/'baseline_validation_8k.json','evaluating_baseline')
        for count in (8,32):
            stage = list(argv);stage[stage.index('--layers')+1]=f'0-{count-1}'
            execute(stage,f'coupled_compile_stage{count}.log',f'constructing_stage{count}')
            assembly = candidate
            if count<32:
                assembly=run/f'coupled_stage{count}_assembly';assembly.mkdir(exist_ok=True)
                for layer in range(32):
                    name=f'compressed_layer_{layer}.pt';path=assembly/name
                    if not path.exists():path.symlink_to((candidate if layer<count else source)/name)
            result=evaluate(assembly,run/f'coupled_stage{count}_validation_8k.json',f'evaluating_stage{count}')
            update(f'stage{count}_measured',child_pid=None,
                   **{f'stage{count}_student':result['student'],f'stage{count}_validation_gates':compare(baseline,result)})
        gates=compare(baseline,result)
        if all(gates.values()):
            confirmation=evaluate(candidate,run/'coupled_full32_heldout_test.json','evaluating_full_test_regression',
                                  split='test',offset=32768,count=1000000000)
            old=json.loads((source/'evaluation_stage32_heldout_test.json').read_text())
            full_gates=compare(old,confirmation)
            update('full_test_measured',child_pid=None,full_test_student=confirmation['student'],full_test_non_degradation=full_gates)
            if all(full_gates.values()):
                execute([python,str(ROOT/'scripts/export_packed_schmidt_model.py'),
                         '--model-dir',model,'--checkpoint-dir',str(candidate),
                         '--quality-report',str(run/'coupled_full32_heldout_test.json'),
                         '--checkpoint-container','gzip','--output',str(run/'coupled_export_gzip')],
                        'coupled_export.log','measuring_actual_complete_package')
                measurement=json.loads((run/'coupled_export_gzip/package_measurement.json').read_text())
                update('package_measured',child_pid=None,actual_package=measurement)
                execute([python,str(ROOT/'scripts/verify_packed_nec_package.py'),
                         '--model-dir',model,'--checkpoint-dir',str(candidate),
                         '--package-dir',str(run/'coupled_export_gzip/deployable_model'),
                         '--output',str(run/'coupled_export_parity.json')],
                        'coupled_export_parity.log','checking_bundled_runtime_parity')
                teacher,student=confirmation['teacher'],confirmation['student']
                final_gates={
                    'storage':measurement['serialized_bytes']<=state['maximum_complete_serialized_bytes'],
                    'structure':measurement['structural_reduction']>=6738417664/state['maximum_complete_logical_coefficients'],
                    'fallback_ppl':student['perplexity']<=teacher['perplexity']+.5,
                    'accuracy':abs(student['next_token_accuracy']-teacher['next_token_accuracy'])<=.02,
                    'agreement':student['dense_top1_agreement']>=.98,
                    'runtime_parity':json.loads((run/'coupled_export_parity.json').read_text())['passed'],
                }
                update('finished',child_pid=None,final_gates=final_gates,
                       dense_ppl_target_met=student['perplexity']<=teacher['perplexity'],
                       final_success_demonstrated=all(final_gates.values()))
            else:update('needs_further_reconstruction',child_pid=None)
        else:update('needs_further_reconstruction',child_pid=None,validation_non_degradation=gates)
    except BaseException:
        update('failed',child_pid=None,error=traceback.format_exc());raise


if __name__=='__main__':main()
