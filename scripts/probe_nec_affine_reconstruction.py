#!/usr/bin/env python3
"""Test sequential analytic mean-error correction at a fixed coefficient cap.

Remove one coordinate amplitude per Down row before adding one FP16 output
bias per row. No gradients are used. This is an affine NEC construction
prototype; package integration and actual byte measurement remain required.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmarks.benchmark_day4_perplexity import inject_nox_checkpoints,sliding_window_specifications
from nox.construction_selection import construction_improves
from nox.data import load_wikitext2_raw_split
from nox.packed_schmidt import reconstruct_packed_schmidt,packed_schmidt_coefficient_count_from_artifact,packed_schmidt_payload_bytes_from_artifact
from nox.residual_codec import codec_payload_bytes


def release_amplitude(artifact):
    a=copy.deepcopy(artifact);rows,columns=a['orig_shape'];k=a['amplitudes_per_row'];codec=a['amplitude_codec']
    mask=np.unpackbits(a['support_bits'].numpy(),bitorder='little',count=rows*columns).reshape(rows,columns)
    last=columns-1-mask[:,::-1].argmax(1);mask[np.arange(rows),last]=0
    a['support_bits']=torch.from_numpy(np.packbits(mask.reshape(-1),bitorder='little'))
    groups=(k-1+codec['group_size']-1)//codec['group_size'];oldpad=codec['groups_per_row']*codec['group_size'];newpad=groups*codec['group_size']
    bits=np.unpackbits(codec['packed_codes'].numpy(),bitorder='little',count=rows*oldpad*codec['bits']).reshape(rows,oldpad,codec['bits'])
    bits=bits[:,:newpad].copy();bits[:,k-1:]=0
    codec.update(shape=(rows,k-1),groups_per_row=groups,
                 packed_codes=torch.from_numpy(np.packbits(bits.reshape(-1),bitorder='little')),
                 group_scales=codec['group_scales'][:,:groups].contiguous().clone())
    codec['payload_bytes']=codec_payload_bytes(codec);a['amplitudes_per_row']=k-1
    a['payload_bytes']=packed_schmidt_payload_bytes_from_artifact(a)
    return a


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','checkpoint-dir','capture-manifest','readout-checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');start=time.monotonic()
    report={'schema':'nec_affine_fixed_coefficient_probe_v1','uses_autograd':False,'final_success_demonstrated':False,'phase':'capturing_teacher'}
    def save():
        (args.output/'status.json').write_text(json.dumps(report,indent=2)+'\n')
    save();capture=json.loads(args.capture_manifest.read_text());context=capture['sequence_length'];offsets=capture['records']['train']['window_offsets'];split=len(offsets)-max(1,len(offsets)//4)
    tokenizer=AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
    train,provenance=load_wikitext2_raw_split('train')
    if provenance['text_sha256']!=capture['records']['train']['source']['text_sha256']:raise ValueError('training corpus mismatch')
    corpus=tokenizer(train,return_tensors='pt',add_special_tokens=True).input_ids[0]
    ids=torch.stack([corpus[o:o+context] for o in offsets]).cuda();labels=ids[split:,1:].reshape(-1)
    model=AutoModelForCausalLM.from_pretrained(args.model_dir,dtype=torch.float16,device_map='auto',local_files_only=True).eval()
    teacher_outputs={}
    def capture_hook(layer):
        def hook(module,inputs,output):teacher_outputs[layer]=(output if torch.is_tensor(output) else output[0]).detach().cpu()
        return hook
    hooks=[block.register_forward_hook(capture_hook(i)) for i,block in enumerate(model.model.layers)]
    teacher_features=model.model(ids,use_cache=False).last_hidden_state[split:,:-1].reshape(-1,model.config.hidden_size)
    for h in hooks:h.remove()
    dense_head=model.lm_head.weight.detach().clone();teacher_logits=F.linear(teacher_features,dense_head).cpu()
    validation,vprovenance=load_wikitext2_raw_split('validation');vid=tokenizer(validation,return_tensors='pt',add_special_tokens=True).input_ids
    windows,limit=sliding_window_specifications(vid.shape[1],seq_len=context,stride=512,max_eval_tokens=8192)
    validation_parts=[]
    for w in windows:
        x=vid[:,w['begin']:w['end']].cuda();target=x.clone()
        if w['target_length']<x.shape[1]:target[:,:-w['target_length']]=-100
        label=target[:,1:];mask=label!=-100
        validation_parts.append((x,mask,label[mask],model(x,use_cache=False).logits[:,:-1][mask].cpu()))
    inject_nox_checkpoints(model,args.checkpoint_dir,expected_layers=list(range(32)))
    model.lm_head.weight.copy_(torch.load(args.readout_checkpoint,map_location='cpu',weights_only=True)['lm_head.weight'].cuda())
    def metrics(logits,teacher,labels):
        nll=kl=0.;correct=agree=0
        for b in range(0,len(labels),256):
            s=logits[b:b+256].float();t=teacher[b:b+256].cuda().float();y=labels[b:b+256]
            slp,tlp=s.log_softmax(-1),t.log_softmax(-1)
            nll+=float(F.nll_loss(slp,y,reduction='sum').double());kl+=float(F.kl_div(slp,tlp,log_target=True,reduction='none').sum(-1).double().sum())
            st,tt=s.argmax(-1),t.argmax(-1);correct+=int((st==y).sum());agree+=int((st==tt).sum())
        n=len(labels);return dict(mean_nll=nll/n,perplexity=math.exp(nll/n),teacher_kl_nats_per_token=kl/n,next_token_correct=correct,dense_agreement_correct=agree,scored_tokens=n,next_token_accuracy=correct/n,dense_top1_agreement=agree/n)
    def guard_score():
        h=model.model(ids[split:],use_cache=False).last_hidden_state[:,:-1].reshape(-1,model.config.hidden_size)
        return metrics(F.linear(h,model.lm_head.weight),teacher_logits,labels)
    def validation_score():
        pieces=[];teachers=[];ys=[]
        for x,mask,y,t in validation_parts:
            pieces.append(model(x,use_cache=False).logits[:,:-1][mask]);teachers.append(t);ys.append(y)
        return metrics(torch.cat(pieces),torch.cat(teachers),torch.cat(ys))
    baseline=guard_score();baseline_validation=validation_score();report.update(phase='constructing',baseline_training_guard=baseline,baseline_validation=baseline_validation);save()
    modified=[];released=0
    for layer,block in enumerate(model.model.layers):
        checkpoint=torch.load(args.checkpoint_dir/f'compressed_layer_{layer}.pt',map_location='cpu',weights_only=True)
        old=checkpoint['compressed_artifacts']['down_proj'];new=release_amplitude(old)
        released+=packed_schmidt_coefficient_count_from_artifact(old)-packed_schmidt_coefficient_count_from_artifact(new)
        block.mlp.down_proj.weight.copy_(reconstruct_packed_schmidt(new,device='cuda',dtype=torch.float16))
        block.mlp.down_proj.bias=torch.nn.Parameter(torch.zeros(block.mlp.down_proj.out_features,device='cuda',dtype=torch.float16),requires_grad=False)
        checkpoint['compressed_artifacts']['down_proj']=new;modified.append(checkpoint)
    biases={}
    def correction_hook(layer):
        def hook(module,inputs,output):
            actual=output if torch.is_tensor(output) else output[0]
            target=teacher_outputs[layer][:split].cuda()
            bias=(target.float()-actual[:split].float()).mean((0,1)).half()
            module.mlp.down_proj.bias.copy_(bias);biases[layer]=bias.clone()
            corrected=actual+bias
            return corrected if torch.is_tensor(output) else (corrected,*output[1:])
        return hook
    hooks=[block.register_forward_hook(correction_hook(i)) for i,block in enumerate(model.model.layers)]
    model.model(ids,use_cache=False)
    for h in hooks:h.remove()
    added=sum(v.numel() for v in biases.values())
    if added>released:raise RuntimeError('bias coefficients exceed released amplitude budget')
    report.update(released_amplitude_and_scale_coefficients=released,added_bias_coefficients=added,trials=[])
    best=baseline;selected=None
    for fraction in (1.,.5):
        for layer,block in enumerate(model.model.layers):block.mlp.down_proj.bias.copy_(biases[layer]*fraction)
        candidate=guard_score();accepted=construction_improves(candidate,best)
        report['trials'].append({'fraction':fraction,'score':candidate,'accepted':accepted})
        if accepted:best=candidate;selected=fraction
        print(json.dumps(report['trials'][-1]),flush=True);save()
    if selected is not None:
        for layer,block in enumerate(model.model.layers):block.mlp.down_proj.bias.copy_(biases[layer]*selected)
        score=validation_score();report.update(selected_fraction=selected,selected_training_guard=best,selected_validation=score,
                                              validation_improves_all_metrics=construction_improves(score,baseline_validation))
        if report['validation_improves_all_metrics']:
            for layer,checkpoint in enumerate(modified):
                checkpoint['compressed_artifacts']['down_proj']['output_bias']=model.model.layers[layer].mlp.down_proj.bias.cpu()
                torch.save(checkpoint,args.output/f'compressed_layer_{layer}.pt')
    report.update(phase='measured',seconds=time.monotonic()-start,validation_used_for_selection=False,deployable_package_not_yet_measured=True)
    save();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
