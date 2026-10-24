from pathlib import Path
import json,subprocess,traceback
root=Path('/tmp/nec-native-preconditioned-source');run=Path('/workspace/runs/nec-accuracy-recovery-20261024')
py='/workspace/venvs/nec-hf/bin/python';baseline=Path('/tmp/nec-regularized-selected32');alternative=Path('/tmp/nec-native-up-branch32');assembly=Path('/tmp/nec-native-up-branch-selected32')
reports=run/'native_up_branch_full32';reports.mkdir(exist_ok=False)
model='/workspace/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d';readout=run/'readout_probe_exact/constructed_readout.pt'
state={'schema':'nec_native_up_branch_full32_v1','final_success_demonstrated':False,'maximum_serialized_bytes':2417355138,'maximum_coefficients':3323032892}
def update(phase,**kw):
 state.update(phase=phase,**kw);p=reports/'status.tmp';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(reports/'status.json')
def execute(argv,name,phase):
 with open(reports/name,'a') as f:
  p=subprocess.Popen(list(map(str,argv)),stdout=f,stderr=subprocess.STDOUT,cwd=root);update(phase,child_pid=p.pid,command=list(map(str,argv)))
  if p.wait():raise RuntimeError(phase+' failed')
def compare(a,b):
 if a['input_token_ids_sha256']!=b['input_token_ids_sha256']:raise RuntimeError('evaluation token mismatch')
 x,y=a['student'],b['student']
 return {**{k:y[k]<=x[k] for k in ['perplexity','teacher_kl_nats_per_token']},**{k:y[k]>=x[k] for k in ['next_token_accuracy','dense_top1_agreement']}}
def evaluate(split,count,path):
 execute([py,root/'scripts/evaluate_packed_teacher_kl.py','--checkpoint-dir',assembly,'--model-id',model,'--readout-checkpoint',readout,'--split',split,'--token-offset','32768' if split=='test' else '0','--max-eval-tokens',count,'--batch-size','1','--output',path],path.stem+'.log','evaluating_'+split)
 return json.loads(path.read_text())
try:
 argv=json.loads((run/'canonical_relative_status.json').read_text())['compiler_argv'];argv[1]=str(root/'scripts/compile_packed_schmidt_model.py')
 for k,v in {'--output':str(alternative),'--layers':'0-31','--reconstruct-from':str(baseline),'--joint-mlp-sweeps':'1','--joint-mlp-cg-iterations':'32','--joint-attention-sweeps':'0','--joint-mlp-output-metric':'energy','--functional-ridge-metric':'diagonal'}.items():argv[argv.index(k)+1]=v
 argv+=['--functional-branch-preconditioner','--reuse-initial-operators','--functional-mlp-roles','up_proj','--functional-mlp-sides','calibration_left','calibration_right','--functional-solve-precision','high']
 execute(argv,'compile.log','constructing_full32')
 execute([py,root/'scripts/select_nec_operator_constructions.py','--model-dir',model,'--baseline-dir',baseline,'--alternative-dir',alternative,'--capture-manifest',Path('/workspace/runs/nec-sparse10-coupled-sequential-full32/activation_capture_manifest.json'),'--report-dir',reports,'--assembly-dir',assembly,'--passes','1','--groups','mlp','--initial-construction','best_guard','--readout-checkpoint',readout],'selection.log','selecting_on_training_guard')
 d=evaluate('validation','8192',reports/'validation_8k.json')
 old=json.loads((run/'regularized_down/validation_8k.json').read_text());gates=compare(old,d);update('validation_measured',gates=gates,quality=d['student'])
 if all(gates.values()) and (d['student']['perplexity']<old['student']['perplexity'] or d['student']['teacher_kl_nats_per_token']<old['student']['teacher_kl_nats_per_token']):
  d=evaluate('test','1000000000',reports/'heldout_test.json')
  old=json.loads((run/'regularized_down/heldout_test.json').read_text());gates=compare(old,d);q=d['student']
  target={'ppl':q['perplexity']<=5.867,'kl':q['teacher_kl_nats_per_token']<=.5,'accuracy':.54<=q['next_token_accuracy']<=.62,'agreement':q['dense_top1_agreement']>=.85}
  update('full_test_measured',gates=gates,quality=q,user_quality_gates=target,all_user_quality_targets_met=all(target.values()),storage_not_yet_measured=True,child_pid=None)
 else:update('rejected_or_no_validation_gain',child_pid=None)
except BaseException:
 update('failed',child_pid=None,error=traceback.format_exc());raise

