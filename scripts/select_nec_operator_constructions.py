#!/usr/bin/env python3
"""Select between two all-NEC constructions using actual teacher logits.

All transformer operators remain compressed at exactly the same tensor budgets.
Search uses only construction-training guard windows; independent validation
and complete deployable-byte measurement must follow this search.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints
from nox.compiler import reconstruct_from_artifact
from nox.construction_selection import construction_improves
from nox.data import load_wikitext2_raw_split
from nox.packed_schmidt import packed_schmidt_coefficient_count_from_artifact, packed_schmidt_payload_bytes_from_artifact

ROLES=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
GROUPS={'attention':ROLES[:4], 'mlp':ROLES[4:]}


def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def modules(model, layer):
    block=model.model.layers[layer]
    return {name:getattr(block.self_attn if name in ROLES[:4] else block.mlp,name) for name in ROLES}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--alternative-dir',type=Path,required=True)
    p.add_argument('--capture-manifest',type=Path,required=True)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--assembly-dir',type=Path,required=True)
    p.add_argument('--passes',type=int,default=2)
    p.add_argument('--initial-construction',choices=('baseline','best_guard'),default='baseline')
    args=p.parse_args()
    args.report_dir.mkdir(parents=True,exist_ok=True)
    if args.assembly_dir.exists():raise FileExistsError(args.assembly_dir)
    if args.passes<1:raise ValueError('search passes must be positive')
    config=json.loads((args.model_dir/'config.json').read_text())
    config_hash=hashlib.sha256((args.model_dir/'config.json').read_bytes()).hexdigest()
    layers=int(config['num_hidden_layers'])
    expected={f'compressed_layer_{i}.pt' for i in range(layers)}
    for directory in (args.baseline_dir,args.alternative_dir):
        if {p.name for p in directory.glob('compressed_layer_*.pt')}!=expected:
            raise ValueError('construction search requires two complete compressed models')
        if json.loads((directory/'model_manifest.json').read_text())['model_config_sha256']!=config_hash:
            raise ValueError('construction source model configuration differs')
    capture=json.loads(args.capture_manifest.read_text())
    if capture['model_config_sha256']!=config_hash:
        raise ValueError('activation capture uses a different model configuration')
    offsets=capture['records']['train']['window_offsets']
    count=max(1,len(offsets)//4);offsets=offsets[-count:]
    context=int(capture['sequence_length'])
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    text,provenance=load_wikitext2_raw_split('train')
    if provenance['text_sha256']!=capture['records']['train']['source']['text_sha256']:
        raise ValueError('construction corpus differs from captured training data')
    corpus=tokenizer(text,return_tensors='pt',add_special_tokens=True).input_ids[0]
    ids=torch.stack([corpus[i:i+context] for i in offsets])
    token_hash=hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    ids=ids.cuda();labels=ids[:,1:];scored=labels.numel()
    torch.set_num_threads(8)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,dtype=torch.float16,
                                             device_map='auto',local_files_only=True).eval()
    teacher=model(ids,use_cache=False).logits[:,:-1].float()
    teacher_predictions=teacher.argmax(-1)
    teacher_logp=teacher.log_softmax(-1);teacher_probability=teacher_logp.exp()
    teacher_nll=float(F.nll_loss(teacher_logp.reshape(-1,teacher.shape[-1]),labels.reshape(-1)))
    del teacher

    def score():
        logits=model(ids,use_cache=False).logits[:,:-1].float()
        predictions=logits.argmax(-1);logp=logits.log_softmax(-1)
        nll=float(F.nll_loss(logp.reshape(-1,logp.shape[-1]),labels.reshape(-1)))
        kl=float((teacher_probability*(teacher_logp-logp)).sum(dtype=torch.float64)/scored)
        correct=int((predictions==labels).sum());agreement=int((predictions==teacher_predictions).sum())
        return dict(mean_nll=nll,perplexity=math.exp(nll) if nll < 709 else None,teacher_kl_nats_per_token=kl,
                    next_token_correct=correct,dense_agreement_correct=agreement,scored_tokens=scored,
                    next_token_accuracy=correct/scored,dense_top1_agreement=agreement/scored)

    inject_nox_checkpoints(model,args.baseline_dir,expected_layers=list(range(layers)))
    current=score()
    state={'schema':'nec_model_level_construction_selection_v1','phase':'searching',
           'baseline_dir':str(args.baseline_dir.resolve()),'alternative_dir':str(args.alternative_dir.resolve()),
           'selection_corpus':provenance,'window_offsets':offsets,'input_token_ids_sha256':token_hash,
           'teacher':{'perplexity':math.exp(teacher_nll),
                      'next_token_accuracy':float((teacher_predictions==labels).float().mean())},
           'baseline':current,'current':current,'trials':[],'selected':{},
           'actual_model_evaluations':1,'all_transformer_layers_remain_nec':True,
           'teacher_model_evaluations':1,
           'validation_used_for_selection':False,'deployable_bytes_not_yet_measured':True,
           'final_success_demonstrated':False}
    state['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__), args.capture_manifest, args.baseline_dir/'model_manifest.json',
        args.alternative_dir/'model_manifest.json', ROOT/'nox/residual_codec.py', ROOT/'nox/packed_schmidt.py')}
    initial=current
    if args.initial_construction == 'best_guard':
        inject_nox_checkpoints(model,args.alternative_dir,expected_layers=list(range(layers)))
        alternative_score=score()
        state['actual_model_evaluations']+=1
        take_alternative=construction_improves(alternative_score,current)
        state['initial_alternative_probe']={
            'score':alternative_score,'accepted':take_alternative,
            'selection_data':'training guard only',
        }
        if take_alternative:
            current=alternative_score
            state['selected']={f'{layer}:{group}':True for layer in range(layers) for group in GROUPS}
            state['current']=current
        else:
            inject_nox_checkpoints(model,args.baseline_dir,expected_layers=list(range(layers)))
    atomic_json(args.report_dir/'selection.json',state)
    started=time.monotonic()
    for sweep in range(args.passes):
        accepted=0
        order=list(range(layers)) if sweep%2==0 else list(reversed(range(layers)))
        for layer in order:
            name=f'compressed_layer_{layer}.pt'
            baseline=torch.load(args.baseline_dir/name,map_location='cpu',weights_only=True)
            alternative=torch.load(args.alternative_dir/name,map_location='cpu',weights_only=True)
            if baseline['layer_idx']!=layer or alternative['layer_idx']!=layer:
                raise ValueError('source checkpoint layer identity differs')
            target_modules=modules(model,layer)
            for group,roles in GROUPS.items():
                key=f'{layer}:{group}';was_alternative=state['selected'].get(key,False)
                target=baseline if was_alternative else alternative
                for role in roles:
                    old,new=baseline['compressed_artifacts'][role],alternative['compressed_artifacts'][role]
                    for metric in (packed_schmidt_coefficient_count_from_artifact,packed_schmidt_payload_bytes_from_artifact):
                        if metric(old)!=metric(new):raise ValueError('constructor alternatives have different tensor budgets')
                original={role:target_modules[role].weight.detach().clone() for role in roles}
                for role in roles:
                    module=target_modules[role]
                    module.weight.copy_(reconstruct_from_artifact(target['compressed_artifacts'][role],device=module.weight.device,dtype=module.weight.dtype))
                proposed=score();state['actual_model_evaluations']+=1
                keep=construction_improves(proposed,current)
                if keep:
                    state['selected'][key]=not was_alternative;current=proposed;accepted+=1
                else:
                    for role in roles:target_modules[role].weight.copy_(original[role])
                del original
                record={'pass':sweep,'layer':layer,'group':group,'proposed_alternative':not was_alternative,
                        'accepted':keep,'score':proposed}
                state['trials'].append(record);state['current']=current
                atomic_json(args.report_dir/'selection.json',state)
                print(json.dumps(record),flush=True)
            del baseline,alternative
        if not accepted:break

    # The persistent selection recipe can rebuild this temporary assembly
    # entirely from the two persistent source checkpoint directories.
    args.assembly_dir.mkdir(parents=True)
    records=[]
    for layer in range(layers):
        name=f'compressed_layer_{layer}.pt'
        baseline=torch.load(args.baseline_dir/name,map_location='cpu',weights_only=True)
        alternative=torch.load(args.alternative_dir/name,map_location='cpu',weights_only=True)
        sources={}
        for group,roles in GROUPS.items():
            selected=state['selected'].get(f'{layer}:{group}',False)
            for role in roles:
                sources[role]='alternative' if selected else 'baseline'
                if selected:baseline['compressed_artifacts'][role]=alternative['compressed_artifacts'][role]
        destination=args.assembly_dir/name
        torch.save(baseline,destination)
        records.append({'layer':layer,'projection_sources':sources,
                        'payload_bytes':sum(packed_schmidt_payload_bytes_from_artifact(a) for a in baseline['compressed_artifacts'].values()),
                        'logical_coefficient_count':sum(packed_schmidt_coefficient_count_from_artifact(a) for a in baseline['compressed_artifacts'].values()),
                        'checkpoint_sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    manifest=json.loads((args.baseline_dir/'model_manifest.json').read_text())
    checkpoint_bytes=sum(p.stat().st_size for p in args.assembly_dir.glob('compressed_layer_*.pt'))
    coefficient_count=sum(r['logical_coefficient_count'] for r in records)
    if coefficient_count!=manifest['logical_artifact_coefficient_count_present']:
        raise RuntimeError('assembled model changed total artifact coefficient count')
    plan=manifest['storage_preflight']
    upper_bound=checkpoint_bytes+plan['fixed_package_bytes']+plan['metadata_reserve_bytes']
    manifest.update(layers=records,construction_method='global_teacher_kl_guarded_nec_construction_selection',
                    construction_selection_input_ids_sha256=token_hash,
                    baseline_construction=str(args.baseline_dir.resolve()),alternative_construction=str(args.alternative_dir.resolve()),
                    serialized_checkpoint_bytes_present=checkpoint_bytes,
                    logical_artifact_coefficient_count_present=coefficient_count,
                    evaluated_assembly_package_bytes_upper_bound=upper_bound,
                    evaluated_assembly_package_ratio_lower_bound=plan['dense_package_bytes']/upper_bound,
                    projected_complete_package_bytes_upper_bound=upper_bound,
                    projected_complete_package_ratio_lower_bound=plan['dense_package_bytes']/upper_bound,
                    projected_storage_budget_satisfied=upper_bound<=plan['package_cap_bytes'],
                    seconds=time.monotonic()-started)
    atomic_json(args.assembly_dir/'model_manifest.json',manifest)
    inject_nox_checkpoints(model,args.assembly_dir,expected_layers=list(range(layers)))
    assembled_score=score();state['actual_model_evaluations']+=1
    if (any(assembled_score[k]!=current[k] for k in ('next_token_correct','dense_agreement_correct'))
            or any(abs(assembled_score[k]-current[k])>1e-7 for k in ('mean_nll','teacher_kl_nats_per_token'))):
        raise RuntimeError('saved assembly differs from the selected model')
    state.update(phase='assembled',assembly_dir=str(args.assembly_dir.resolve()),seconds=time.monotonic()-started,
                 strict_training_guard_improvement=construction_improves(assembled_score,initial),
                 saved_assembly_score_verified=True, current=assembled_score)
    atomic_json(args.report_dir/'selection.json',state)
    print(json.dumps({'baseline':initial,'selected':current,'actual_model_evaluations':state['actual_model_evaluations']}),flush=True)


if __name__=='__main__':main()
