#!/usr/bin/env python3
"""Measure analytical teacher-hidden readout reconstruction for an NEC body.

This explicitly reconstructs the existing FP16 output head by a prior-anchored
linear solve. It uses no gradients or extra inference coefficients. It is an
analytical distillation experiment, not an untouched-head NEC evaluation.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints, sliding_window_specifications
from nox.construction_selection import construction_improves
from nox.data import load_wikitext2_raw_split


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'checkpoint-dir', 'capture-manifest', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    # Decoder arithmetic must match the reference evaluator exactly. Reduced
    # precision is allowed only for the temporary statistical linear solve.
    torch.set_float32_matmul_precision('highest')
    start = time.monotonic()
    report = {'schema':'nec_analytical_readout_probe_v1', 'uses_autograd':False,
              'output_head_reconstructed':True, 'inference_coefficients_added':0,
              'body_checkpoint_dir':str(args.checkpoint_dir), 'phase':'capturing_teacher',
              'validation_used_for_selection':False, 'final_success_demonstrated':False}
    def save():
        temporary=args.output/'status.tmp'
        temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(args.output/'status.json')
    save()
    capture=json.loads(args.capture_manifest.read_text())
    config_hash=hashlib.sha256((args.model_dir/'config.json').read_bytes()).hexdigest()
    if config_hash!=capture['model_config_sha256']:
        raise ValueError('capture/model configuration mismatch')
    context=int(capture['sequence_length'])
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    train, train_provenance=load_wikitext2_raw_split('train')
    if train_provenance['text_sha256']!=capture['records']['train']['source']['text_sha256']:
        raise ValueError('training corpus does not match capture')
    train_ids=tokenizer(train,return_tensors='pt',add_special_tokens=True).input_ids[0]
    offsets=capture['records']['train']['window_offsets']
    batches=[train_ids[i:i+context][None,:] for i in offsets]
    validation, validation_provenance=load_wikitext2_raw_split('validation')
    validation_ids=tokenizer(validation,return_tensors='pt',add_special_tokens=True).input_ids
    windows, limit=sliding_window_specifications(validation_ids.shape[1],seq_len=context,stride=512,max_eval_tokens=8192)
    validation_batches=[validation_ids[:,w['begin']:w['end']] for w in windows]
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,dtype=torch.float16,
                                            device_map='auto',local_files_only=True).eval()
    head=model.lm_head.weight.detach().clone()
    def features(parts):
        result=[]
        for begin in range(0,len(parts),4):
            # Capturing model.model skips the large vocabulary projection.
            group=parts[begin:begin+4]
            if len({x.shape[1] for x in group})!=1:
                for x in group:
                    result.append(model.model(x.cuda(),use_cache=False).last_hidden_state[0].cpu())
            else:
                hidden=model.model(torch.cat(group).cuda(),use_cache=False).last_hidden_state.cpu()
                result.extend(hidden.unbind(0))
        return result
    teacher_train=features(batches)
    teacher_validation=features(validation_batches)
    report.update(phase='capturing_nec');save()
    inject_nox_checkpoints(model,args.checkpoint_dir,expected_layers=list(range(32)))
    student_train=features(batches)
    student_validation=features(validation_batches)
    def join_train(hidden, first, last):
        x=torch.cat([h[:-1] for h in hidden[first:last]]).cuda()
        labels=torch.cat([p[0,1:] for p in batches[first:last]]).cuda()
        return x,labels
    split=len(batches)-max(1,len(batches)//4)
    x, _=join_train(student_train,0,split)
    y, _=join_train(teacher_train,0,split)
    guard_x,labels=join_train(student_train,split,len(batches))
    guard_teacher,_=join_train(teacher_train,split,len(batches))
    def teacher_logits(h):
        return F.linear(h,head).float()
    guard_logits=teacher_logits(guard_teacher)
    def score(h, candidate_head, teacher, labels):
        nll=kl=0.;correct=agree=teacher_correct=0
        for begin in range(0,len(h),256):
            end=begin+256
            t=teacher[begin:end].float();s=F.linear(h[begin:end],candidate_head).float()
            tlp=t.log_softmax(-1);slp=s.log_softmax(-1);label=labels[begin:end]
            nll+=float(F.nll_loss(slp,label,reduction='sum').double())
            kl+=float((tlp.exp()*(tlp-slp)).sum(dtype=torch.float64))
            prediction=s.argmax(-1);tp=t.argmax(-1)
            correct+=int((prediction==label).sum());agree+=int((prediction==tp).sum())
            teacher_correct+=int((tp==label).sum())
        n=len(h)
        return {'mean_nll':nll/n,'perplexity':math.exp(nll/n),
                'teacher_kl_nats_per_token':kl/n,'next_token_correct':correct,
                'dense_agreement_correct':agree,'scored_tokens':n,
                'next_token_accuracy':correct/n,'dense_top1_agreement':agree/n,
                'teacher_next_token_accuracy':teacher_correct/n}
    baseline=score(guard_x,head,guard_logits,labels)
    best=baseline;best_head=head
    report.update(phase='solving',baseline_training_guard=baseline,trials=[]);save()
    torch.set_float32_matmul_precision('high')
    xf,yf=x.float(),y.float()
    covariance=xf.T@xf/len(x)
    cross=(yf-xf).T@xf/len(x)
    diagonal=covariance.diagonal().clamp_min(1e-8)
    for ridge in (.01,.1,1.,10.):
        gram=covariance+torch.diag(ridge*diagonal)
        delta=torch.cholesky_solve(cross.T,torch.linalg.cholesky(gram)).T
        correction=head.float()@delta
        for fraction in (.25,.5,1.):
            candidate=(head.float()+fraction*correction).half()
            measured=score(guard_x,candidate,guard_logits,labels)
            accepted=construction_improves(measured,best)
            trial={'ridge':ridge,'fraction':fraction,'accepted':accepted,'score':measured}
            report['trials'].append(trial)
            if accepted:
                best=measured;best_head=candidate
                report['selected']={'ridge':ridge,'fraction':fraction}
            print(json.dumps(trial),flush=True);save()
    report['selected_training_guard']=best
    def join_validation(hidden):
        selected=[];targets=[]
        for h,p,w in zip(hidden,validation_batches,windows):
            target=p.clone()
            if w['target_length']<p.shape[1]:target[:,:-w['target_length']]=-100
            label=target[:,1:][0];mask=label!=-100
            selected.append(h[:-1][mask]);targets.append(label[mask])
        return torch.cat(selected).cuda(),torch.cat(targets).cuda()
    vx,vlabels=join_validation(student_validation)
    vt,_=join_validation(teacher_validation)
    vlogits=teacher_logits(vt)
    report['baseline_validation']=score(vx,head,vlogits,vlabels)
    report['selected_validation']=score(vx,best_head,vlogits,vlabels)
    report['validation_improves_all_metrics']=construction_improves(
        report['selected_validation'],report['baseline_validation'])
    report['validation_provenance']=validation_provenance
    report['validation_input_token_ids_sha256']=hashlib.sha256(validation_ids[:,:limit].numpy().tobytes()).hexdigest()
    report['training_window_offsets']=offsets
    report['fit_windows']=split;report['guard_windows']=len(batches)-split
    report['head_coefficient_count']=head.numel()
    report['phase']='measured';report['seconds']=time.monotonic()-start
    if report['validation_improves_all_metrics']:
        torch.save({'lm_head.weight':best_head.cpu()},args.output/'constructed_readout.pt')
    save();print(json.dumps({k:v for k,v in report.items() if k!='trials'}),flush=True)


if __name__=='__main__':main()
