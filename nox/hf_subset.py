"""Fetch only the first N decoder layers of a safetensors-sharded HF model via HTTP Range."""

import json
import struct
import urllib.request
from pathlib import Path

import torch

_DTYPES = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def _get(url, byte_range=None):
    headers = {"User-Agent": "NoxResearch/1.0"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        return r.read()


def fetch_layer_prefix(repo, revision, n_layers, out_path):
    """Save embed_tokens + layers[0:n_layers] as one safetensors file; returns its path."""
    from safetensors.torch import save_file
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    base = f"https://huggingface.co/{repo}/resolve/{revision}"
    weight_map = json.loads(_get(f"{base}/model.safetensors.index.json"))["weight_map"]
    wanted = [k for k in weight_map if k == "model.embed_tokens.weight" or any(
        k.startswith(f"model.layers.{i}.") for i in range(n_layers))]
    headers, tensors = {}, {}
    for key in wanted:
        shard = weight_map[key]
        if shard not in headers:
            n = struct.unpack("<Q", _get(f"{base}/{shard}", (0, 7)))[0]
            headers[shard] = (n, json.loads(_get(f"{base}/{shard}", (8, 7 + n))))
        n, header = headers[shard]
        meta = header[key]
        lo, hi = meta["data_offsets"]
        raw = bytearray(_get(f"{base}/{shard}", (8 + n + lo, 8 + n + hi - 1)))
        tensors[key] = torch.frombuffer(raw, dtype=_DTYPES[meta["dtype"]]).reshape(meta["shape"]).clone()
        print(f"fetched {key} {tuple(meta['shape'])}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    save_file(tensors, str(tmp))
    tmp.rename(out_path)
    return out_path
