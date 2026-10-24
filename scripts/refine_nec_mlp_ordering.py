#!/usr/bin/env python3
"""Jointly construct the identity/permuted SwiGLU screen in its own gauge."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from nox.block_compiler import scaled_dot_product_attention_2d
from nox.functional_schmidt import construct_functional_schmidt_mlp
from nox.packed_schmidt import reconstruct_packed_schmidt,packed_schmidt_coefficient_count_from_artifact,packed_schmidt_payload_bytes_from_artifact
from nox.sparse_schmidt import construction_matmul_precision
from scripts.compile_packed_schmidt_model import SafetensorReader,layer_tensor_names,dense_projection_inputs,rms_norm


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','checkpoint-dir','activation-dir','source-screen','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gate-target-policy',choices=('teacher_preactivation','readout_projection'),default='teacher_preactivation')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');start=time.monotonic()
    config=json.loads((args.model_dir/'config.json').read_text());heads=config['num_attention_heads'];eps=config['rms_norm_eps'];split=24576
    source={k:v.cuda().float() for k,v in SafetensorReader(args.model_dir).load(layer_tensor_names(0)).items()}
    hidden=torch.load(args.activation_dir/'calibration_input_layer_0.pt',map_location='cuda',weights_only=True).float()
    teacher=dense_projection_inputs(hidden,source,layer=0,sequence_length=2048,n_heads=heads,n_kv_heads=heads,eps=eps)
    checkpoint=torch.load(args.checkpoint_dir/'compressed_layer_0.pt',map_location='cpu',weights_only=True)
    original={k:reconstruct_packed_schmidt(a,device='cuda') for k,a in checkpoint['compressed_artifacts'].items()}
    prefix='model.layers.0.';n1=rms_norm(hidden,source[prefix+'input_layernorm.weight'],eps)
    context=scaled_dot_product_attention_2d(*(original[r]@n1 for r in ('q_proj','k_proj','v_proj')),heads,heads,2048)
    residual=hidden+original['o_proj']@context;x=rms_norm(residual,source[prefix+'post_attention_layernorm.weight'],eps)
    desired=hidden+source[prefix+'self_attn.o_proj.weight']@teacher['o_proj']+source[prefix+'mlp.down_proj.weight']@teacher['down_proj']-residual
    clean_gate=source[prefix+'mlp.gate_proj.weight']@teacher['gate_proj']
    def errors(w):
        actual=w['down_proj']@(F.silu(w['gate_proj']@x)*(w['up_proj']@x))
        return {'fit_relative_mse':float((actual[:,:split]-desired[:,:split]).square().sum()/desired[:,:split].square().sum()),
                'external_guard_relative_mse':float((actual[:,split:]-desired[:,split:]).square().sum()/desired[:,split:].square().sum())}
    report={'schema':'nec_joint_mlp_ordering_screen_v1','uses_autograd':False,'fit_tokens':split,
            'external_guard_tokens':hidden.shape[1]-split,'external_guard_used_in_construction':False,
            'baseline':errors(original),'candidates':{},'deployable_package_measured':False}
    report['gate_target_policy']=args.gate_target_policy
    for basis in ('identity','neuron_spectral'):
        began=time.monotonic();recipe=torch.load(args.source_screen/(basis+'_experimental.pt'),map_location='cpu',weights_only=True)
        order=recipe['neuron_order'];target=clean_gate[:,:split]
        if order is not None:target=target[order.cuda().long()]
        with construction_matmul_precision('high'):
            artifacts,_,diagnostic=construct_functional_schmidt_mlp(recipe['compressed_artifacts'],x[:,:split],
                desired[:,:split],gate_targets=target,sweeps=1,cg_iterations=8,ridge_ratio=.05,
                sequence_length=2048,output_metric='energy',ridge_metric='mean',
                gate_target_policy=args.gate_target_policy)
        candidate={k:reconstruct_packed_schmidt(a,device='cuda') for k,a in artifacts.items()}
        if order is not None:
            inverse=order.cuda().long().argsort()
            candidate={k:(w[:,inverse] if k=='down_proj' else w[inverse]) for k,w in candidate.items()}
        recipe['compressed_artifacts']=artifacts
        torch.save(recipe,args.output/(basis+'_experimental.pt'))
        torch.save({k:w.half().cpu() for k,w in candidate.items()},args.output/(basis+'_measurement_weights.pt'))
        record={'errors':errors(candidate),'seconds':time.monotonic()-began,'functional_construction':diagnostic,
                'coefficients':sum(packed_schmidt_coefficient_count_from_artifact(a) for a in artifacts.values()),
                'tensor_payload_bytes':sum(packed_schmidt_payload_bytes_from_artifact(a) for a in artifacts.values())+(0 if order is None else order.numel()*2)}
        report['candidates'][basis]=record;report['seconds']=time.monotonic()-start
        (args.output/'screen.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'basis':basis,'errors':record['errors'],'seconds':record['seconds']}),flush=True)


if __name__=='__main__':main()
