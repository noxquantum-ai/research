#!/usr/bin/env python3
"""Screen exact MLP basis changes before another full NEC compilation.

Builds identity, shared-neuron-order and signed-Hadamard versions with the same
MPO/sparse representation and per-operator coefficient ceilings. Outputs are
experimental transformed-basis artifacts, not a deployable package. External
guard tokens never enter their construction. No gradients are used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nox.block_compiler import scaled_dot_product_attention_2d, ridge_compensated_up_target
from nox.packed_schmidt import reconstruct_packed_schmidt, packed_schmidt_coefficient_count_from_artifact, packed_schmidt_payload_bytes_from_artifact
from nox.sparse_schmidt import compile_sparse_schmidt, construction_matmul_precision
from scripts.compile_packed_schmidt_model import SafetensorReader, layer_tensor_names, dense_projection_inputs, rms_norm
from scripts.probe_nec_swiglu_ordering import contribution_gram, spectral_order


def hadamard(values, block=128):
    shape=values.shape
    if shape[-1]%block:raise ValueError('Hadamard blocks must divide the input dimension')
    result=values.reshape(*shape[:-1],-1,block).clone()
    stride=1
    while stride<block:
        pairs=result.reshape(*result.shape[:-1],-1,2,stride)
        a,b=pairs[...,0,:].clone(),pairs[...,1,:].clone()
        pairs[...,0,:]=a+b;pairs[...,1,:]=a-b
        result=pairs.reshape_as(result);stride*=2
    return result.reshape(shape)/(block**.5)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','checkpoint-dir','activation-dir','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    start=time.monotonic();layer=0;split=24576;roles=('gate_proj','up_proj','down_proj')
    checkpoint=torch.load(args.checkpoint_dir/'compressed_layer_0.pt',map_location='cpu',weights_only=True)
    config=json.loads((args.model_dir/'config.json').read_text());sequence=2048;heads=config['num_attention_heads'];eps=config['rms_norm_eps']
    source={k:v.cuda().float() for k,v in SafetensorReader(args.model_dir).load(layer_tensor_names(0)).items()}
    hidden=torch.load(args.activation_dir/'calibration_input_layer_0.pt',map_location='cuda',weights_only=True).float()
    teacher=dense_projection_inputs(hidden,source,layer=0,sequence_length=sequence,n_heads=heads,n_kv_heads=heads,eps=eps)
    prefix='model.layers.0.'
    decoded={k:reconstruct_packed_schmidt(a,device='cuda') for k,a in checkpoint['compressed_artifacts'].items()}
    n1=rms_norm(hidden,source[prefix+'input_layernorm.weight'],eps)
    context=scaled_dot_product_attention_2d(*(decoded[r]@n1 for r in ('q_proj','k_proj','v_proj')),heads,heads,sequence)
    residual=hidden+decoded['o_proj']@context
    x=rms_norm(residual,source[prefix+'post_attention_layernorm.weight'],eps)
    weights={r:source[prefix+'mlp.'+r+'.weight'] for r in roles}
    clean_gate=weights['gate_proj']@teacher['gate_proj'];clean_up=weights['up_proj']@teacher['up_proj']
    desired=hidden+source[prefix+'self_attn.o_proj.weight']@teacher['o_proj']+weights['down_proj']@teacher['down_proj']-residual
    def errors(w):
        actual=w['down_proj']@(F.silu(w['gate_proj']@x)*(w['up_proj']@x))
        return {'fit_relative_mse':float((actual[:,:split]-desired[:,:split]).square().sum()/desired[:,:split].square().sum()),
                'external_guard_relative_mse':float((actual[:,split:]-desired[:,split:]).square().sum()/desired[:,split:].square().sum())}
    report={'schema':'nec_mlp_basis_screen_v1','layer':0,'uses_autograd':False,'fit_tokens':split,
            'external_guard_tokens':hidden.shape[1]-split,'guard_used_for_construction':False,
            'baseline':errors(decoded),'candidates':{},'deployable_package_measured':False,
            'final_success_demonstrated':False}
    def save():
        t=args.output/'screen.tmp';t.write_text(json.dumps(report,indent=2)+'\n');t.replace(args.output/'screen.json')
    save()
    with construction_matmul_precision('high'):
        gram=contribution_gram(teacher['down_proj'][:,:8192],weights['down_proj'])
        order,_=spectral_order(gram)
    del gram
    inverse=order.argsort();generator=torch.Generator().manual_seed(271828)
    signs={r:(torch.randint(0,2,(weights[r].shape[1],),generator=generator)*2-1).cuda().float() for r in roles}
    # Gate and Up must see the same transformed input in this controlled test.
    signs['up_proj']=signs['gate_proj']
    for basis in ('identity','neuron_spectral','signed_hadamard'):
        began=time.monotonic();artifacts={};candidate={};construction={}
        for role in roles:
            source_w=weights[role]
            if role=='gate_proj':inputs=x[:,:split];target=clean_gate[:,:split]
            elif role=='up_proj':
                inputs=x[:,:split]
                target,_,_=ridge_compensated_up_target(clean_gate[:,:split],clean_up[:,:split],
                    candidate['gate_proj']@inputs,down_weight=weights['down_proj'],ridge_ratio=.05)
            else:
                inputs=F.silu(candidate['gate_proj']@x[:,:split])*(candidate['up_proj']@x[:,:split])
                target=desired[:,:split]
            if basis=='neuron_spectral':
                source_w=source_w[order] if role!='down_proj' else source_w[:,order]
                if role!='down_proj':target=target[order]
                else:inputs=inputs[order]
            elif basis=='signed_hadamard':
                source_w=hadamard(source_w*signs[role][None,:])
                inputs=hadamard((inputs*signs[role][:,None]).T).T
            original=checkpoint['compressed_artifacts'][role]
            byte_cap=packed_schmidt_payload_bytes_from_artifact(original)
            coefficient_cap=packed_schmidt_coefficient_count_from_artifact(original)
            metadata_bytes=0 if basis=='identity' else (order.numel()*2 if basis=='neuron_spectral' else 16)
            with construction_matmul_precision('high'):
                artifact,_,diagnostic=compile_sparse_schmidt(source_w,byte_budget=byte_cap-metadata_bytes,
                    coefficient_budget=coefficient_cap,calibration_inputs=inputs,calibration_targets=target,
                    backbone_fraction=.10,amplitude_bits=3,excitation_sequence_length=sequence,
                    bits_candidates=(5,),group_size=64,als_sweeps=3,activation_alpha=.5,
                    calibration_branch_fraction=.20,calibration_branch_bits=5,
                    calibration_branch_fit_tokens=512,column_branch_fraction=.05,column_branch_bits=5,
                    projection_name=role,layout_policy='packed_efficiency',n_heads=heads,head_dim=128,
                    seed=roles.index(role)+4)
            physical=reconstruct_packed_schmidt(artifact,device='cuda')
            if basis=='neuron_spectral':physical=physical[inverse] if role!='down_proj' else physical[:,inverse]
            elif basis=='signed_hadamard':physical=hadamard(physical)*signs[role][None,:]
            candidate[role]=physical;artifacts[role]=artifact
            construction[role]={'coefficient_count':packed_schmidt_coefficient_count_from_artifact(artifact),
                'maximum_coefficients':coefficient_cap,'tensor_payload_plus_basis_metadata':packed_schmidt_payload_bytes_from_artifact(artifact)+metadata_bytes,
                'maximum_tensor_payload':byte_cap,'diagnostics':diagnostic}
            if construction[role]['coefficient_count']>coefficient_cap:raise RuntimeError('coefficient cap exceeded')
        record={'errors':errors(candidate),'construction':construction,'seconds':time.monotonic()-began}
        recipe={'basis':basis,'layer_idx':0,'compressed_artifacts':artifacts,
                'neuron_order':order.cpu().short() if basis=='neuron_spectral' else None,
                'signed_hadamard_seed':271828 if basis=='signed_hadamard' else None,'hadamard_block':128}
        torch.save(recipe,args.output/(basis+'_experimental.pt'))
        # A temporary materialized operator is for measurement only; it is
        # explicitly excluded from any deployable-compression claim.
        torch.save({r:w.half().cpu() for r,w in candidate.items()},args.output/(basis+'_measurement_weights.pt'))
        report['candidates'][basis]=record;report['seconds']=time.monotonic()-start;save()
        print(json.dumps({'basis':basis,'errors':record['errors'],'seconds':record['seconds']}),flush=True)


if __name__=='__main__':main()
