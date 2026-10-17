#!/usr/bin/env python3
"""One-layer, equal-bytes comparison of RTN / GPTQ / incoherent GPTQ with large calibration.

Streams Hessians over many calibration tokens through the real HF model, snapshots them
at several token counts (testing the calibration-size hypothesis), and scores every
projection on held-out dense activations. No gradients, no healing.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nox.data import load_wikitext2_raw_split
from nox.quant import (HessianAccumulator, bits_per_weight, gptq_quantize,
                       incoherent_gptq_quantize, rtn_quantize)

# projection -> (module name within the decoder layer, shared-input key)
PROJECTIONS = {
    "q_proj": "self_attn.q_proj", "k_proj": "self_attn.k_proj", "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj", "gate_proj": "mlp.gate_proj", "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}


class _Stop(Exception):
    pass


def token_windows(tokenizer, split, n_tokens, seq_len, seed):
    text, _ = load_wikitext2_raw_split(split)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_win = n_tokens // seq_len
    starts = torch.randperm((len(ids) - seq_len) // seq_len, generator=torch.Generator().manual_seed(seed))[:n_win]
    return torch.stack([ids[s * seq_len:(s + 1) * seq_len] for s in starts])


@torch.no_grad()
def collect(model, layer_idx, windows, device, hess=None, snapshots=(), keep=None):
    """Run windows through layers 0..layer_idx, updating Hessians and/or keeping samples."""
    layer = model.model.layers[layer_idx]
    state = {"tokens": 0}
    modules = {name: layer.get_submodule(path) for name, path in PROJECTIONS.items()}
    snaps = {}

    def make_hook(name):
        def hook(_, args):
            x = args[0].reshape(-1, args[0].shape[-1])
            if hess is not None and name in hess:
                hess[name].update(x)
            if keep is not None and name in keep:
                keep[name].append(x.float().cpu())
        return hook

    handles = [modules[n].register_forward_pre_hook(make_hook(n)) for n in modules]
    def stop(*_):
        raise _Stop
    handles.append(layer.register_forward_hook(stop))
    try:
        for i, win in enumerate(windows):
            try:
                model(win[None].to(device), use_cache=False)
            except _Stop:
                pass
            state["tokens"] += win.numel()
            if hess is not None and state["tokens"] in snapshots:
                snaps[state["tokens"]] = {n: h.hessian().clone() for n, h in hess.items()}
    finally:
        for h in handles:
            h.remove()
    return snaps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--revision", default="8efe6c9b93655b934e27bd9981e3ec13e55aee9d")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--device", default="auto", help="model device: auto|cuda|mps|cpu")
    p.add_argument("--quant-device", default="cpu", help="device for GPTQ/scoring (cpu is float32-safe)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--weights-file", type=Path, default=Path("models/layer_prefix_15.safetensors"))
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--calib-tokens", type=int, nargs="+", default=[2048, 16384, 131072])
    p.add_argument("--val-tokens", type=int, default=16384)
    p.add_argument("--bits", type=int, nargs="+", default=[4, 3, 2])
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--methods", nargs="+", default=["rtn", "gptq", "incoherent_gptq"])
    p.add_argument("--projections", nargs="+", default=list(PROJECTIONS))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM
    try:
        from transformers.initialization import no_init_weights
    except ImportError:  # transformers < 5
        from transformers.modeling_utils import no_init_weights
    from safetensors.torch import load_file
    from nox.hf_subset import fetch_layer_prefix
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, cache_dir=args.cache_dir)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, cache_dir=args.cache_dir)
    config.num_hidden_layers = args.layer + 1   # layers beyond the target are never needed
    weights = fetch_layer_prefix(args.model, args.revision, args.layer + 1, args.weights_file)
    with no_init_weights():
        torch.set_default_dtype(torch.float16)
        model = LlamaForCausalLM(config)
        torch.set_default_dtype(torch.float32)
    state = {k: v for k, v in load_file(str(weights)).items() if not k.endswith("inv_freq")}
    missing = model.load_state_dict(state, strict=False)
    assert all(k.startswith(("lm_head", "model.norm")) for k in missing.missing_keys), missing
    assert not missing.unexpected_keys, missing
    model = model.to(args.device).eval()
    layer = model.model.layers[args.layer]

    t0 = time.time()
    calib = token_windows(tok, "train", max(args.calib_tokens), args.seq_len, seed=42)
    valid = token_windows(tok, "validation", args.val_tokens, args.seq_len, seed=43)
    dims = {n: layer.get_submodule(path).in_features for n, path in PROJECTIONS.items()}
    hess = {n: HessianAccumulator(dims[n], device=args.device) for n in args.projections}
    qdev = args.quant_device
    snaps = collect(model, args.layer, calib, args.device, hess=hess, snapshots=set(args.calib_tokens))
    keep = {n: [] for n in args.projections}
    collect(model, args.layer, valid, args.device, keep=keep)
    val_x = {n: torch.cat(v).to(qdev) for n, v in keep.items()}
    print(f"collected in {time.time() - t0:.0f}s", flush=True)

    results = {"layer": args.layer, "group_size": args.group_size, "rows": []}
    for name in args.projections:
        w = layer.get_submodule(PROJECTIONS[name]).weight.data.float().to(qdev)
        ref = val_x[name] @ w.T
        for bits in args.bits:
            bpw = bits_per_weight(bits, args.group_size)
            for method in args.methods:
                for n_tok in ([None] if method == "rtn" else args.calib_tokens):
                    t1 = time.time()
                    if method == "rtn":
                        q = rtn_quantize(w, bits, args.group_size)
                    else:
                        h = snaps[n_tok][name].to(qdev)
                        fn = gptq_quantize if method == "gptq" else incoherent_gptq_quantize
                        q = fn(w, h, bits=bits, group_size=args.group_size)
                    err = float((val_x[name] @ q.T - ref).norm() / ref.norm())
                    row = {"projection": name, "bits": bits, "bits_per_weight": bpw,
                           "method": method, "calib_tokens": n_tok, "heldout_rel_err": err,
                           "seconds": time.time() - t1}
                    results["rows"].append(row)
                    print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
