from pathlib import Path
import gzip,json,os,shutil,subprocess,time,hashlib
run=Path('/workspace/runs/nec-accuracy-recovery-20261024');statusdir=run/'native_candidate_boolq';statusdir.mkdir(exist_ok=True)
package=Path('/tmp/nec-boolq-native-ready/deployable_model')
assembly=Path('/tmp/nec-native-up-branch-selected32')
def save_status(obj):
 tmp=statusdir/'status.tmp';tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(statusdir/'status.json')
state={'schema':'nec_selected_full32_boolq_v1','phase':'waiting_for_full_test','final_success_demonstrated':False,'previous_boolq_percent':48.50152905198777,'sample_count':3270}
save_status(state)
while True:
 s=json.loads((run/'native_up_branch_full32/status.json').read_text())
 if s.get('child_pid') is None and s.get('phase') in ('full_test_measured','rejected_full_test','failed','failed_full_test'):break
 time.sleep(5)
if s.get('phase')!='full_test_measured':raise RuntimeError('full heldout PPL run did not complete successfully: '+s.get('phase','?'))
state.update(phase='installing_selected_checkpoints',ppl_result=s)
save_status(state)
hashes=[]
for layer in range(32):
 src=assembly/f'compressed_layer_{layer}.pt'
 dst=package/'nec_checkpoints'/f'compressed_layer_{layer}.pt.gz'
 tmp=dst.with_suffix('.gz.partial')
 with src.open('rb') as fin,tmp.open('wb') as raw:
  with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=6,mtime=0) as zout:shutil.copyfileobj(fin,zout,4*1024*1024)
 os.replace(tmp,dst)
 hashes.append({'layer':layer,'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'bytes':dst.stat().st_size})
serialized=sum(p.stat().st_size for p in package.rglob('*') if p.is_file())
state.update(phase='running_full_boolq',candidate_package=str(package),candidate_package_bytes=serialized,
 whole_model_compression_ratio=13479247511/serialized,checkpoint_hashes=hashes)
save_status(state)
cmd=['/workspace/venvs/nec-hf/bin/python','/workspace/research/benchmarks/evaluate_nec_full_suite.py',
 '--package-dir',str(package),'--output-dir',str(statusdir/'harness'),'--cache-dir','/tmp/nec-boolq-cache-native',
 '--batch-size','auto:4','--max-batch-size','32','--tasks','boolq']
with (statusdir/'boolq.log').open('w') as f:
 p=subprocess.run(cmd,cwd='/workspace/research',stdout=f,stderr=subprocess.STDOUT)
if p.returncode:raise RuntimeError(f'full BoolQ evaluation exited {p.returncode}')
result=json.loads((statusdir/'harness'/'boolq_summary.json').read_text());score=result['score_percent']
state.update(phase='complete',boolq=result,beats_previous=score>state['previous_boolq_percent'],delta_percentage_points=score-state['previous_boolq_percent'])
save_status(state)
