#!/usr/bin/env python3
"""Select experimental layer-0 MLP functions on training logits, then validate.

Materialized candidate weights are temporary measurements of saved NEC
transformed-basis artifacts. This tool makes no deployable-byte claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints,sliding_window_specifications
from nox.construction_selection import construction_improves
from nox.data import load_wikitext2_raw_split


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','checkpoint-dir','capture-manifest','readout-checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate',type=Path,nargs='+',required=True)
    args=p.parse_args();torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    capture=json.loads(args.capture_manifest.read_text());context=capture['sequence_length']
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    train,provenance=load_wikitext2_raw_split('train')
    if provenance['text_sha256']!=capture['records']['train']['source']['text_sha256']:
        raise ValueError('construction training corpus changed')
    corpus=tokenizer(train,return_tensors='pt',add_special_tokens=True).input_ids
    offsets=capture['records']['train']['window_offsets'];offsets=offsets[-max(1,len(offsets)//4):]
    guard=[(corpus[:,o:o+context],context) for o in offsets]
    validation,vprovenance=load_wikitext2_raw_split('validation')
    vid=tokenizer(validation,return_tensors='pt',add_special_tokens=True).input_ids
    windows,limit=sliding_window_specifications(vid.shape[1],seq_len=context,stride=512,max_eval_tokens=8192)
    validation=[(vid[:,w['begin']:w['end']],w['target_length']) for w in windows]
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,dtype=torch.float16,device_map='auto',local_files_only=True).eval()
    def teacher(parts):
        result=[]
        for ids,length in parts:
            targets=ids.clone()
            if length<ids.shape[1]:targets[:,:-length]=-100
            labels=targets[:,1:];mask=labels!=-100
            logits=model(ids.cuda(),use_cache=False).logits[:,:-1][mask].cpu()
            result.append((mask,labels[mask],logits))
        return result
    tg,tv=teacher(guard),teacher(validation)
    inject_nox_checkpoints(model,args.checkpoint_dir,expected_layers=list(range(32)))
    readout=torch.load(args.readout_checkpoint,map_location='cpu',weights_only=True)
    model.lm_head.weight.copy_(readout['lm_head.weight'].cuda())
    def score(parts,teachers):
        nll=kl=0.;correct=agreement=count=0
        for (ids,_),(mask,labels,dense) in zip(parts,teachers):
            logits=model(ids.cuda(),use_cache=False).logits[:,:-1][mask.cuda()].float()
            for begin in range(0,len(labels),256):
                s=logits[begin:begin+256];t=dense[begin:begin+256].cuda().float();label=labels[begin:begin+256].cuda()
                slp,tlp=s.log_softmax(-1),t.log_softmax(-1)
                nll+=float(F.nll_loss(slp,label,reduction='sum').double())
                kl+=float(F.kl_div(slp,tlp,log_target=True,reduction='none').sum(-1).double().sum())
                st,tt=s.argmax(-1),t.argmax(-1)
                correct+=int((st==label).sum());agreement+=int((st==tt).sum());count+=len(label)
        return {'perplexity':math.exp(nll/count),'mean_nll':nll/count,'teacher_kl_nats_per_token':kl/count,
                'next_token_correct':correct,'dense_agreement_correct':agreement,'scored_tokens':count,
                'next_token_accuracy':correct/count,'dense_top1_agreement':agreement/count}
    baseline=score(guard,tg);baseline_validation=score(validation,tv)
    report={'schema':'nec_transformed_mlp_model_guard_v1','selection_split':'construction train guard only',
            'validation_used_for_selection':False,'baseline_training_guard':baseline,
            'baseline_validation':baseline_validation,'trials':[],'selected':None,
            'guard_window_offsets':offsets,'validation_provenance':vprovenance,
            'validation_input_ids_sha256':hashlib.sha256(vid[:,:limit].numpy().tobytes()).hexdigest(),
            'deployable_model_not_yet_verified':True,'final_success_demonstrated':False}
    best=baseline;selected=None;modules={r:getattr(model.model.layers[0].mlp,r) for r in ('gate_proj','up_proj','down_proj')}
    original={r:m.weight.detach().clone() for r,m in modules.items()}
    for path in args.candidate:
        weights=torch.load(path,map_location='cpu',weights_only=True)
        if set(weights)!=set(modules):raise ValueError('candidate must replace exactly Gate/Up/Down')
        for role,module in modules.items():
            if weights[role].shape!=module.weight.shape or weights[role].dtype!=module.weight.dtype or not torch.isfinite(weights[role]).all():
                raise ValueError('candidate shape, dtype or finiteness mismatch')
            module.weight.copy_(weights[role].cuda())
        measured=score(guard,tg);accepted=construction_improves(measured,best)
        report['trials'].append({'candidate':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'training_guard':measured,'accepted':accepted})
        if accepted:best=measured;selected=path
        print(json.dumps(report['trials'][-1]),flush=True)
    if selected:
        weights=torch.load(selected,map_location='cpu',weights_only=True)
        for r,m in modules.items():m.weight.copy_(weights[r].cuda())
        result=score(validation,tv)
        report.update(selected=str(selected),selected_training_guard=best,selected_validation=result,
                      validation_improves_all_metrics=construction_improves(result,baseline_validation))
    else:
        for r,m in modules.items():m.weight.copy_(original[r])
        report['validation_improves_all_metrics']=False
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
