#!/usr/bin/env python3
import hashlib, json, os, sys, time
from pathlib import Path

PKG = Path('/tmp/nec-final-allocation-32layer/deployable_model')
OUT = Path('/tmp/nec-public-benchmarks-32layer')
REPO = Path('/workspace/research-kl-diagnostic')
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(REPO))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
import datasets
from datasets import load_dataset as _unused
from huggingface_hub import dataset_info
from transformers import AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from load_nec_package import load_model
from nox.evaluation import MMLU_REVISION
from nox.mmlu import load_full_mmlu

MMLU_REVISION_EXPECTED = 'c30699e8356da336a370243923dbaf21066bb9fe'
HELLASWAG_REVISION = '218ec52e09a7e7462a5400043bb9a69a41d06b76'
if MMLU_REVISION != MMLU_REVISION_EXPECTED:
    raise RuntimeError(f'Unexpected pinned MMLU revision: {MMLU_REVISION}')

orig_load_dataset = datasets.load_dataset
dataset_calls = []
def pinned_load_dataset(path, *args, **kwargs):
    if path == 'cais/mmlu':
        supplied = kwargs.get('revision')
        if supplied not in (None, MMLU_REVISION_EXPECTED):
            raise ValueError(f'Unexpected MMLU revision requested: {supplied}')
        kwargs['revision'] = MMLU_REVISION_EXPECTED
        dataset_calls.append({'dataset': path, 'revision': MMLU_REVISION_EXPECTED})
    elif path == 'Rowan/hellaswag':
        supplied = kwargs.get('revision')
        if supplied not in (None, HELLASWAG_REVISION):
            raise ValueError(f'Unexpected HellaSwag revision requested: {supplied}')
        kwargs['revision'] = HELLASWAG_REVISION
        dataset_calls.append({'dataset': path, 'revision': HELLASWAG_REVISION})
    return orig_load_dataset(path, *args, **kwargs)
datasets.load_dataset = pinned_load_dataset

OUT.mkdir(parents=True, exist_ok=False)
# Audit full MMLU source and all 57 subjects before loading the model.
mmlu_splits, mmlu_data = load_full_mmlu()
if len(mmlu_splits['test']) != 14042 or len({r['subject'] for r in mmlu_splits['test']}) != 57:
    raise RuntimeError('Pinned MMLU test split was incomplete')
# Pin and record the HellaSwag dataset revision used by the benchmark task.
hella_info = dataset_info('Rowan/hellaswag')
if hella_info.sha != HELLASWAG_REVISION:
    raise RuntimeError(f'HellaSwag dataset moved: expected {HELLASWAG_REVISION}, got {hella_info.sha}')

print('loading selected deployable NEC package', flush=True)
load_started = time.monotonic()
from transformers import AutoModelForCausalLM
_original_from_pretrained = AutoModelForCausalLM.from_pretrained

def _sdpa_from_pretrained(cls, *args, **kwargs):
    kwargs.setdefault('attn_implementation', 'sdpa')
    return _original_from_pretrained(*args, **kwargs)
AutoModelForCausalLM.from_pretrained = classmethod(_sdpa_from_pretrained)
model = load_model(device_map='auto', torch_dtype=torch.float16).eval()
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(PKG, local_files_only=True, use_fast=True)
if len(model.model.layers) != 32:
    raise RuntimeError('Expected complete 32-layer Llama architecture')
print(f'model loaded in {time.monotonic()-load_started:.1f}s; gpu_mem={torch.cuda.memory_allocated()/2**30:.2f}GiB', flush=True)
lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=32, max_length=4096, logits_cache=False)
common = dict(
    model=lm,
    batch_size=32,
    device='cuda:0',
    bootstrap_iters=0,
    check_integrity=False,
    log_samples=False,
    apply_chat_template=False,
    fewshot_as_multiturn=False,
    random_seed=0,
    numpy_random_seed=1234,
    torch_random_seed=1234,
    fewshot_random_seed=1234,
)

# Harness adapter smoke: one MMLU question per subject and a few HellaSwag examples.
print('running MMLU adapter smoke', flush=True)
smoke = evaluator.simple_evaluate(tasks=['mmlu'], num_fewshot=5, limit=1, **common)
if smoke is None or len([k for k in smoke['results'] if k.startswith('mmlu_')]) < 57:
    raise RuntimeError('MMLU smoke did not cover all subjects')
print('MMLU adapter smoke passed for all 57 subjects', flush=True)
print('running HellaSwag adapter smoke', flush=True)
smoke_h = evaluator.simple_evaluate(tasks=['hellaswag'], num_fewshot=0, limit=8, **common)
if smoke_h is None or 'hellaswag' not in smoke_h.get('results', {}):
    raise RuntimeError('HellaSwag smoke failed')
print('HellaSwag adapter smoke passed', flush=True)

# Full MMLU: official Llama 2 comparison is 5-shot, 14,042 test questions.
print('starting full MMLU 5-shot (14,042 test questions)', flush=True)
t0 = time.monotonic()
mmlu = evaluator.simple_evaluate(tasks=['mmlu'], num_fewshot=5, limit=None, **common)
if mmlu is None:
    raise RuntimeError('MMLU evaluator returned no results')
mmlu_results = mmlu.get('results', {})
mmlu_subjects = {k: v for k, v in mmlu_results.items() if k.startswith('mmlu_')}
if len(mmlu_subjects) != 57:
    raise RuntimeError(f'Expected 57 MMLU subjects, got {len(mmlu_subjects)}')
mmlu_counts = mmlu.get('n-samples', {})
counts = [mmlu_counts.get(k, {}).get('effective', mmlu_counts.get(k, {}).get('original')) for k in mmlu_subjects]
if any(not isinstance(n, int) for n in counts) or sum(counts) != 14042:
    raise RuntimeError(f'MMLU did not evaluate all 14,042 test examples: {sum(n for n in counts if isinstance(n,int))}')
mmlu_acc = mmlu.get('groups', {}).get('mmlu', {}).get('acc,none')
if not isinstance(mmlu_acc, (int, float)):
    mmlu_acc = sum(mmlu_results[k]['acc,none'] * mmlu_counts[k].get('effective', mmlu_counts[k].get('original')) for k in mmlu_subjects) / 14042
mmlu_record = {
    'status': 'COMPLETE', 'task': 'mmlu', 'shots': 5, 'split': 'test', 'questions': 14042, 'batch_size': 32,
    'subjects': 57, 'accuracy': float(mmlu_acc), 'accuracy_percent': 100*float(mmlu_acc),
    'published_meta_llama2_7b_reference_percent': 45.3,
    'published_reference': 'Llama 2 paper, Table 19, 5-shot MMLU',
    'reference_url': 'https://arxiv.org/abs/2307.09288',
    'dataset_revision': MMLU_REVISION_EXPECTED,
    'task_version': mmlu.get('versions', {}).get('mmlu'),
    'task_subject_accuracies': {k: v.get('acc,none') for k,v in mmlu_subjects.items()},
    'seconds': time.monotonic()-t0,
}
(OUT/'mmlu_5shot.json').write_text(json.dumps(mmlu_record, indent=2)+'\n')
print(json.dumps({'MMLU_COMPLETE':mmlu_record['accuracy_percent'],'questions':14042,'seconds':mmlu_record['seconds']}), flush=True)

# Full HellaSwag: use zero-shot normalized choice likelihood (acc_norm).
print('starting full HellaSwag 0-shot (validation split)', flush=True)
t0 = time.monotonic()
hella = evaluator.simple_evaluate(tasks=['hellaswag'], num_fewshot=0, limit=None, **common)
if hella is None or 'hellaswag' not in hella.get('results', {}):
    raise RuntimeError('HellaSwag evaluator returned no results')
h_metrics = hella['results']['hellaswag']
h_counts = hella.get('n-samples', {}).get('hellaswag', {})
h_count = h_counts.get('effective', h_counts.get('original'))
if h_count != 10042:
    raise RuntimeError(f'Expected all 10,042 HellaSwag validation examples, got {h_count}')
hella_record = {
    'status':'COMPLETE', 'task':'hellaswag', 'shots':0, 'split':'validation', 'questions':h_count, 'batch_size': 32,
    'accuracy':h_metrics.get('acc,none'), 'accuracy_percent':100*h_metrics['acc,none'],
    'normalized_accuracy':h_metrics.get('acc_norm,none'),
    'normalized_accuracy_percent':100*h_metrics['acc_norm,none'],
    'published_meta_llama2_7b_reference_percent':77.2,
    'published_reference':'Llama 2 paper, Table 20, 0-shot HellaSwag',
    'reference_url':'https://arxiv.org/abs/2307.09288',
    'dataset_revision':HELLASWAG_REVISION,
    'task_version':hella.get('versions',{}).get('hellaswag'),
    'seconds':time.monotonic()-t0,
}
(OUT/'hellaswag_zeroshot.json').write_text(json.dumps(hella_record,indent=2)+'\n')
print(json.dumps({'HELLASWAG_COMPLETE':hella_record['normalized_accuracy_percent'],'questions':h_count,'seconds':hella_record['seconds']}),flush=True)

allocation = json.loads(Path('/tmp/nec-final-allocation-32layer/report.json').read_text())
summary = {
  'status':'COMPLETE',
  'model':'NousResearch/Llama-2-7b-hf',
  'model_revision':allocation['model_revision'],
  'package_path':str(PKG),
  'allocation':allocation['selected_allocation'],
  'allocation_report_sha256':hashlib.sha256(Path('/tmp/nec-final-allocation-32layer/report.json').read_bytes()).hexdigest(),
  'benchmark_harness':{'name':'EleutherAI/lm-evaluation-harness','version':'0.4.13','git_commit':'ddd67220430a2470529f25fd5c05a576ca1057a0'},
  'datasets_loaded':dataset_calls,
  'mmlu_source':mmlu_data,
  'mmlu':mmlu_record,
  'hellaswag':hella_record,
  'interpretation':'Meta reported these scores using its internal evaluation library. Harness scores are useful public-protocol comparisons, not exact reproduction of Meta internal evaluation.',
  'total_benchmark_runtime_seconds':time.monotonic()-load_started,
}
(OUT/'benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('BENCHMARKS_COMPLETE '+str(OUT/'benchmark_summary.json'),flush=True)
