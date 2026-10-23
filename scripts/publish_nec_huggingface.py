#!/usr/bin/env python3
"""Publish a verified NEC package directly from the GPU pod.

Credentials are read from a hidden prompt or an ephemeral worker environment.
They are never written to the repository, package, command line or status file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def file_sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_model_card(run_dir, repo_id):
    package = run_dir / 'export_gzip/deployable_model'
    measurement_path = run_dir / 'export_gzip/package_measurement.json'
    measurement = json.loads(measurement_path.read_text())
    quality = json.loads((run_dir / 'evaluation_stage32_heldout_test.json').read_text())
    parity = json.loads((run_dir / 'export_gzip/parity.json').read_text())
    status = json.loads((run_dir / 'status.json').read_text())
    if not status['success_demonstrated'] or not parity['passed']:
        raise RuntimeError('full-model quality, storage, structure and runtime checks must pass before upload')
    if quality['compressed_layers'] != list(range(32)):
        raise RuntimeError('this release requires all 32 layers')
    if not all(row['lossless_verified'] for row in measurement['lossless_container_verification']):
        raise RuntimeError('lossless checkpoint round-trip checks are incomplete')
    verification = package / 'verification'
    verification.mkdir(exist_ok=True)
    shutil.copy2(run_dir / 'export_gzip/parity.json', verification / 'runtime_parity.json')
    for name in ('representation_breakdown.json', 'paired_greedy_examples.json',
                 'ablation_main_mpo_validation_4k.json'):
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, verification / name)
    license_url = 'https://raw.githubusercontent.com/meta-llama/llama/main/LICENSE'
    with urllib.request.urlopen(license_url, timeout=60) as response:
        license_bytes = response.read()
    license_text = license_bytes.decode('utf-8')
    if 'LLAMA 2 COMMUNITY LICENSE AGREEMENT' not in license_text or 'July 18, 2023' not in license_text:
        raise RuntimeError('the source license is not the expected Llama 2 agreement')
    (package / 'LICENSE.txt').write_bytes(license_bytes)
    (package / 'NOTICE').write_text(
        'Llama 2 is licensed under the LLAMA 2 Community License, '
        'Copyright (c) Meta Platforms, Inc. All Rights Reserved.\n'
    )
    write_json(verification / 'base_license_provenance.json', {
        'source_url': license_url, 'sha256': hashlib.sha256(license_bytes).hexdigest(),
        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
    })
    packages = ('torch', 'transformers', 'huggingface_hub', 'safetensors', 'accelerate', 'numpy')
    requirements = []
    for name in packages:
        version = importlib.metadata.version(name).split('+')[0]
        requirements.append(f'{name}=={version}')
    try:
        requirements.append('sentencepiece==' + importlib.metadata.version('sentencepiece'))
    except importlib.metadata.PackageNotFoundError:
        pass
    (package / 'requirements.txt').write_text('\n'.join(requirements) + '\n')
    t, s = quality['teacher'], quality['student']
    readme = f'''---
license: llama2
language:
- en
base_model: NousResearch/Llama-2-7b-hf
library_name: pytorch
pipeline_tag: text-generation
inference: false
tags:
- nec
- nox-quantum
- quantum-inspired
- tensor-network
- structural-compression
- training-free
- custom-code
---

# NEC Llama 2 7B Hybrid

Research checkpoint produced by the Nox Entanglement Compiler (NEC).
The complete original 32-layer Llama-2-7B architecture is retained. All seven
attention/MLP projections in every layer use a Schmidt MPO backbone plus
sparse residual amplitudes. Embeddings, output head and normalization tensors
remain FP16. Construction uses no gradient-based fine-tuning or healing.

Source model: `NousResearch/Llama-2-7b-hf`, revision
`8efe6c9b93655b934e27bd9981e3ec13e55aee9d`. Llama 2 was developed by Meta.

## Measured results

| Metric | Dense source | NEC hybrid |
| --- | ---: | ---: |
| Serialized model | 13.479 GB | approximately {measurement['serialized_bytes']/1e9:.3f} GB |
| Storage compression | 1x | approximately {measurement['compression_ratio']:.3f}x |
| Stored weight/factor coefficients, including scales and all branches | 6,738,417,664 | {measurement['complete_model_coefficient_count']:,} |
| Structural coefficient reduction | 1x | {measurement['structural_reduction']:.3f}x |
| Held-out perplexity | {t['perplexity']:.6f} | {s['perplexity']:.6f} |
| Teacher KL, nats/token | 0 | {s['teacher_kl_nats_per_token']:.6f} |
| Next-token accuracy | {100*t['next_token_accuracy']:.3f}% | {100*s['next_token_accuracy']:.3f}% |
| Dense top-1 agreement | 100% | {100*s['dense_top1_agreement']:.3f}% |
| Dense transformer layers retained | 32 | 0 |

Quality is measured on {quality['scored_tokens']:,} scored WikiText-2 raw test
tokens after token offset {quality['token_offset']:,}, with context 2048 and
stride 512. Calibration uses 32768 training tokens in 2048-token windows.
This checkpoint meets the declared PPL <=10, storage >=5x and coefficient
reduction >=2x targets together. It still has a substantial quality gap from
the dense teacher. Dense-equivalent task accuracy and 10x compression are
not demonstrated. The example completions are qualitative checks, not a
downstream benchmark score.

The gzip containers are lossless and preserve the original NEC checkpoint
bytes exactly. They improve serialized storage only; coefficient counts and
model predictions are unchanged. Package measurements include retained
weights, compressed artifacts, masks, scales, loader code and metadata.
Integer coordinate metadata and deterministic padding are excluded from
coefficient counts but included in storage bytes.

## Load the complete model

Use the bundled loader to reconstruct all NEC projections. Loading only the
retained safetensors with the ordinary Transformers constructor leaves the
NEC projections uninitialized.

```python
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import sys

path = snapshot_download("{repo_id}")  # authenticate for a private repository
sys.path.insert(0, path)
from load_nec_package import load_model

tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = load_model(device_map="auto")
inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Tested runtime versions are in `requirements.txt`. The loader reconstructs
dense inference matrices. These storage savings do not imply equivalent
runtime VRAM or latency reductions. The generic Hub safetensors parameter
counter may count only the retained tensors; the correct complete stored
coefficient count includes the NEC files and is **{measurement['complete_model_coefficient_count']:,}**.

## Reproducibility and scope

The bundled decoder passed a separate-interpreter test with bitwise-equal
logits against the evaluated checkpoint assembly. Each gzip file has a
verified source/round-trip SHA-256 pair in `nec_allocation_manifest.json`.
Raw held-out metrics are in `nec_quality_evaluation.json`; component counts
and qualitative examples, where present, are in `verification/`.

NEC is explicitly a tensor-network/sparsity hybrid. Conventional sparsity
and packing are not claimed as novel quantum methods. Component ablations
zero the main MPO cores without reoptimizing the remaining representation;
they are contribution checks, not matched reoptimized sparse baselines or
evidence of quantum advantage. Generalization to other base models has not
yet been demonstrated by this experiment.

Compiler source and experiment reports:
https://github.com/noxquantum-ai/research/tree/main/results/nec_sparse_excitation_a40_20261023

Base-model terms and attribution are included in `LICENSE.txt` and `NOTICE`.
'''
    (package / 'README.md').write_text(readme)
    sizes = {str(p.relative_to(package)): p.stat().st_size for p in package.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    measurement.update(serialized_bytes=sum(sizes.values()), file_sizes=sizes, file_count=len(sizes),
                       compression_ratio=measurement['source_model_serialized_bytes'] / sum(sizes.values()))
    measurement['acceptance']['storage_passed'] = measurement['compression_ratio'] >= 5
    if not measurement['acceptance']['storage_passed']:
        raise RuntimeError('complete publication package exceeds the storage cap')
    write_json(measurement_path, measurement)
    return package, measurement, quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--public', action='store_true', help='Default: create a private model repository.')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    token = os.environ.pop('HF_TOKEN', None) or getpass.getpass('HF token: ')
    os.environ['HF_HOME'] = '/tmp/nec-hf-publication-cache'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    state_path = args.run_dir / 'hf_upload_status.json'
    state = {'repo_id': args.repo_id, 'url': 'https://huggingface.co/' + args.repo_id,
             'requested_private': not args.public}

    def update(phase, **extra):
        state.update(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(state_path, state)

    try:
        if not args.worker:
            identity = api.whoami()
            if args.repo_id.split('/')[0] != identity['name']:
                raise ValueError('this publisher requires the authenticated personal namespace')
            api.create_repo(repo_id=args.repo_id, repo_type='model', private=not args.public, exist_ok=False)
            update('repository_created', authenticated_account=identity['name'])
            argv = [sys.executable, str(Path(__file__).resolve()), '--run-dir', str(args.run_dir),
                    '--repo-id', args.repo_id, '--worker'] + (['--public'] if args.public else [])
            environment = {**os.environ, 'HF_TOKEN': token}
            fd = os.open(args.run_dir / 'hf_upload.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, 'a') as log:
                process = subprocess.Popen(argv, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'repo_id': args.repo_id, 'private': not args.public, 'worker_pid': process.pid}))
            return
        update('waiting_for_verified_package')
        parity_path = args.run_dir / 'export_gzip/parity.json'
        while not parity_path.is_file():
            time.sleep(10)
        update('preparing_model_card')
        package, measurement, quality = prepare_model_card(args.run_dir, args.repo_id)
        update('uploading', package_bytes=measurement['serialized_bytes'])
        commit = api.upload_folder(repo_id=args.repo_id, repo_type='model', folder_path=package,
                                   commit_message='Publish verified full-32-layer NEC hybrid checkpoint',
                                   ignore_patterns=['**/__pycache__/**', '*.pyc', '.cache/**'])
        update('verifying_remote_files', revision=commit.oid)
        info = api.model_info(args.repo_id, revision=commit.oid, files_metadata=True)
        remote_sizes, verified_lfs = {}, []
        for item in info.siblings:
            lfs = item.lfs
            lfs_size = (lfs.get('size') if isinstance(lfs, dict) else getattr(lfs, 'size', None)) if lfs else None
            actual_size = lfs_size if lfs_size is not None else item.size
            if actual_size is None:
                raise RuntimeError('Hub file-size metadata is incomplete')
            remote_sizes[item.rfilename] = actual_size
            local = package / item.rfilename
            if local.is_file():
                if local.stat().st_size != actual_size:
                    raise RuntimeError(f'uploaded size mismatch: {item.rfilename}')
                digest = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
                if digest:
                    if file_sha256(local) != digest:
                        raise RuntimeError(f'uploaded hash mismatch: {item.rfilename}')
                    verified_lfs.append(item.rfilename)
        required = {str(p.relative_to(package)) for p in package.rglob('*')
                    if p.is_file() and '__pycache__' not in p.parts}
        if not required.issubset(remote_sizes):
            raise RuntimeError('some deployable model files are absent from the uploaded commit')
        total = sum(remote_sizes.values())
        report = {
            'repo_id': args.repo_id, 'url': state['url'], 'revision': info.sha, 'private': info.private,
            'actual_repository_file_bytes': total, 'file_sizes': remote_sizes,
            'source_fp16_serialized_bytes': measurement['source_model_serialized_bytes'],
            'whole_model_compression_ratio': measurement['source_model_serialized_bytes'] / total,
            'complete_model_coefficient_count': measurement['complete_model_coefficient_count'],
            'structural_reduction': measurement['structural_reduction'],
            'heldout_scored_tokens': quality['scored_tokens'], 'heldout_ppl': quality['student']['perplexity'],
            'all_local_model_files_present_with_matching_sizes': True,
            'lfs_files_verified_by_sha256': verified_lfs,
            'transport_metadata_also_counted': True,
        }
        report['three_targets_passed'] = (report['heldout_ppl'] <= 10
                                           and report['whole_model_compression_ratio'] >= 5
                                           and report['structural_reduction'] >= 2)
        if not report['three_targets_passed']:
            raise RuntimeError('the uploaded repository does not satisfy all three targets')
        write_json(args.run_dir / 'hf_repository_measurement.json', report)
        update('complete', revision=info.sha, private=info.private,
               actual_repository_file_bytes=total,
               whole_model_compression_ratio=report['whole_model_compression_ratio'],
               structural_reduction=report['structural_reduction'],
               three_targets_passed=True)
    except BaseException as error:
        sanitized = str(error).replace(token, '[redacted]')
        update('failed', error_type=type(error).__name__, error=sanitized)
        print(type(error).__name__ + ': ' + sanitized, file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
