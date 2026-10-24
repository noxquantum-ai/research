from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,time,traceback,gzip,hashlib,shutil
root=Path('/workspace/research');run=Path('/workspace/runs/nec-accuracy-recovery-20261024')
py='/workspace/venvs/nec-hf/bin/python'
baseline=Path('/workspace/runs/nec-sparse10-coupled-sequential-full32')
alternative=run/'coupled-full32';assembly=Path('/tmp/nec-selected-construction32')
reports=run/'logit_selection';reports.mkdir(exist_ok=True)
model='/workspace/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d'
state={'schema':'nec_global_construction_selection_queue_v1','final_success_demonstrated':False,'baseline_maximum_bytes':2490273288,'baseline_maximum_coefficients':3323032892}
def update(phase,**extra):
 state.update(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
 tmp=reports/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(reports/'status.json')
def execute(command,name,phase):
 update(phase,command=command)
 with open(reports/name,'a') as f:
  p=subprocess.Popen(command,cwd=root,stdout=f,stderr=subprocess.STDOUT);update(phase,child_pid=p.pid)
  if p.wait():raise RuntimeError(f'{phase} failed; see {name}')
def evaluate(split,offset,count,destination):
 execute([py,str(root/'scripts/evaluate_packed_teacher_kl.py'),'--checkpoint-dir',str(assembly),'--model-id',model,'--split',split,'--token-offset',str(offset),'--max-eval-tokens',str(count),'--output',str(destination)],destination.stem+'.log','evaluating_'+split)
 return json.loads(destination.read_text())
def compare(a,b):
 if a['input_token_ids_sha256']!=b['input_token_ids_sha256']:raise RuntimeError('evaluation token IDs mismatch')
 x,y=a['student'],b['student']
 return {**{k:y[k]<=x[k] for k in ('perplexity','teacher_kl_nats_per_token')},**{k:y[k]>=x[k] for k in ('next_token_accuracy','dense_top1_agreement')}}
try:
 update('waiting_for_sequential_trial')
 while True:
  phase=json.loads((run/'coupled_status.json').read_text())['phase']
  if phase in ('finished','needs_further_reconstruction'):break
  if phase=='failed':raise RuntimeError('sequential source trial failed')
  time.sleep(15)
 if len(list(alternative.glob('compressed_layer_*.pt')))!=32:raise RuntimeError('alternative is not full32')
 execute([py,str(root/'scripts/select_nec_operator_constructions.py'),'--model-dir',model,'--baseline-dir',str(baseline),'--alternative-dir',str(alternative),'--capture-manifest',str(baseline/'activation_capture_manifest.json'),'--report-dir',str(reports),'--assembly-dir',str(assembly)],'selection.log','selecting_teacher_logit_constructions')
 selected=evaluate('validation',0,8192,reports/'validation_8k.json')
 old=json.loads((run/'baseline_validation_8k.json').read_text());gates=compare(old,selected)
 update('validation_measured',child_pid=None,validation_quality=selected['student'],validation_non_degradation=gates)
 if all(gates.values()):
  final=evaluate('test',32768,1000000000,reports/'heldout_test.json')
  old=json.loads((baseline/'evaluation_stage32_heldout_test.json').read_text());fullgates=compare(old,final)
  update('full_test_measured',child_pid=None,full_test_quality=final['student'],full_test_non_degradation=fullgates)
  if all(fullgates.values()):
   output=Path('/tmp/nec-selected-export-gzip')
   execute([py,str(root/'scripts/export_packed_schmidt_model.py'),'--model-dir',model,'--checkpoint-dir',str(assembly),'--quality-report',str(reports/'heldout_test.json'),'--checkpoint-container','gzip','--output',str(output)],'export.log','measuring_complete_package')
   measurement=json.loads((output/'package_measurement.json').read_text());package=Path(measurement['path'])
   records=[]
   if measurement['serialized_bytes']>2490273288:
    update('testing_lossless_container_level9')
    for file in sorted((package/'nec_checkpoints').glob('*.gz')):
     old=file.read_bytes();raw=gzip.decompress(old);new=gzip.compress(raw,compresslevel=9,mtime=0)
     if len(new)<len(old):
      if gzip.decompress(new)!=raw:raise RuntimeError('container recompression changed decoded bytes')
      tmp=file.with_suffix('.tmp');tmp.write_bytes(new);tmp.replace(file)
     records.append({'file':file.name,'before':len(old),'after':file.stat().st_size,'raw_sha256':hashlib.sha256(raw).hexdigest()})
    allocation_path=package/'nec_allocation_manifest.json'
    allocation=json.loads(allocation_path.read_text())
    by_name={r['file']:r for r in records}
    for record in allocation.get('lossless_container_verification',[]):
     r=by_name[record['file']]
     if record['uncompressed_sha256']!=r['raw_sha256']:raise RuntimeError('recompression source hash mismatch')
     record.update(container_bytes=r['after'],compression_level=9 if r['after']<r['before'] else 1)
    allocation_path.write_text(json.dumps(allocation,indent=2)+'\n')
    sizes={str(p.relative_to(package)):p.stat().st_size for p in package.rglob('*') if p.is_file()}
    measurement.update(serialized_bytes=sum(sizes.values()),file_sizes=sizes,file_count=len(sizes),lossless_level9_trials=records,lossless_container_verification=allocation.get('lossless_container_verification',[]))
    measurement['compression_ratio']=measurement['source_model_serialized_bytes']/measurement['serialized_bytes']
   (reports/'package_measurement.json').write_text(json.dumps(measurement,indent=2)+'\n')
   execute([py,str(root/'scripts/verify_packed_nec_package.py'),'--model-dir',model,'--checkpoint-dir',str(assembly),'--package-dir',str(package),'--output',str(reports/'runtime_parity.json')],'runtime_parity.log','verifying_exported_runtime')
   t,s=final['teacher'],final['student']
   target={'storage':measurement['serialized_bytes']<=2490273288,'structure':measurement['complete_model_coefficient_count']<=3323032892,'fallback_ppl':s['perplexity']<=t['perplexity']+.5,'accuracy':abs(s['next_token_accuracy']-t['next_token_accuracy'])<=.02,'agreement':s['dense_top1_agreement']>=.98,'runtime_parity':json.loads((reports/'runtime_parity.json').read_text())['passed']}
   measurement['acceptance']={'maximum_complete_serialized_bytes':2490273288,'maximum_complete_coefficients':3323032892,'maximum_ppl_excess_over_dense':.5,'maximum_accuracy_gap_points':2,'minimum_dense_top1_agreement':.98,'gates':target,'all_numerical_targets_met':all(target.values()),'persistent_copy_required_before_promotion':True}
   (reports/'package_measurement.json').write_text(json.dumps(measurement,indent=2)+'\n')
   (output/'package_measurement.json').write_text(json.dumps(measurement,indent=2)+'\n')
   update('measured',child_pid=None,final_gates=target,ideal_dense_ppl_met=s['perplexity']<=t['perplexity'],package_in_scratch=str(package),requires_persistent_package_copy_before_promotion=True,final_success_demonstrated=False,all_numerical_targets_met=all(target.values()))
  else:update('needs_further_accuracy_work',child_pid=None)
 else:update('needs_further_accuracy_work',child_pid=None)
except BaseException:
 update('failed',child_pid=None,error=traceback.format_exc());raise
