"""Day 3 Benchmark: Multi-Layer Sequential Cascade Test (Step 4.2 Inter-Layer Causal Healing).

Evaluates cascading layer-by-layer compression across a sequence of Transformer layers
(e.g. Layers 15 -> 18 or full Layers 0 -> 31). Proves that calibrating downstream layers on
actual upstream compressed activations stabilizes residual stream drift compared to unhealed
Multiverse and independent matrix compression.
"""

import argparse
import hashlib
import json
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc
import struct
import sys
import time
import urllib.request
from pathlib import Path
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    k, v = stripped.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v

import torch.nn.functional as F

HF_TOKEN = os.getenv("HF_TOKEN")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nox.sequential_compiler import compile_sequential_cascade, get_tier_config, forward_transformer_block
from nox.utils import compute_metrics
from nox.artifacts import checkpoint_file_sizes, fingerprint, source_files_fingerprint
from nox.data import load_wikitext2_raw_split


def log_step(msg):
    """Prints timestamped progress log with immediate stdout flush."""
    t_str = time.strftime("%H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


def load_cached_tensor(cache_path):
    """Return a valid tensor cache entry, or None so the caller can refetch it."""
    if not os.path.isfile(cache_path):
        return None
    if os.path.getsize(cache_path) == 0:
        log_step(f"Ignoring empty tensor cache '{cache_path}'; it will be fetched again.")
        return None
    try:
        tensor = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        log_step(f"Ignoring unreadable tensor cache '{cache_path}' ({exc}); it will be fetched again.")
        return None
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        log_step(f"Ignoring invalid tensor cache '{cache_path}'; it will be fetched again.")
        return None
    return tensor.to(dtype=torch.float32)


def fetch_layer_weights(
    repo_id="NousResearch/Llama-2-7b-hf",
    layer_idx=15,
    save_dir="./models/Llama-2-7b-hf",
    token=None,
    revision="main",
):
    """
    Fetches all 7 projection matrices and 2 layer norms for a single Transformer layer
    directly via HTTP Range requests without downloading the full model.
    Caches tensors locally in save_dir.
    """
    os.makedirs(save_dir, exist_ok=True)
    tensor_mapping = {
        "q_proj": f"model.layers.{layer_idx}.self_attn.q_proj.weight",
        "k_proj": f"model.layers.{layer_idx}.self_attn.k_proj.weight",
        "v_proj": f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        "o_proj": f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        "gate_proj": f"model.layers.{layer_idx}.mlp.gate_proj.weight",
        "up_proj": f"model.layers.{layer_idx}.mlp.up_proj.weight",
        "down_proj": f"model.layers.{layer_idx}.mlp.down_proj.weight",
        "input_layernorm": f"model.layers.{layer_idx}.input_layernorm.weight",
        "post_attention_layernorm": f"model.layers.{layer_idx}.post_attention_layernorm.weight",
    }

    layer_weights = {}
    missing_tensors = {}

    for name, tensor_key in tensor_mapping.items():
        cache_path = os.path.join(save_dir, f"layer_{layer_idx}_{name}.pt")
        cached_tensor = load_cached_tensor(cache_path)
        if cached_tensor is not None:
            log_step(f"Loading cached {name} (Layer {layer_idx}) from '{cache_path}'...")
            layer_weights[name] = cached_tensor
        else:
            missing_tensors[name] = tensor_key

    if not missing_tensors:
        log_step(f"All Layer {layer_idx} weights loaded from local cache.")
        return layer_weights

    headers = {"User-Agent": "Mozilla/5.0 (compatible; NoxResearch/1.0)"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    log_step(f"Resolving safetensors shard index for '{repo_id}' to fetch missing Layer {layer_idx} tensors...")
    index_url = f"https://huggingface.co/{repo_id}/raw/{revision}/model.safetensors.index.json"
    req_index = urllib.request.Request(index_url, headers=headers)
    with urllib.request.urlopen(req_index) as resp:
        index_json = json.loads(resp.read().decode("utf-8"))
    weight_map = index_json.get("weight_map", {})

    shard_headers = {}

    for name, tensor_key in missing_tensors.items():
        shard_name = weight_map.get(tensor_key, "model-00001-of-00002.safetensors")
        file_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{shard_name}"

        if shard_name not in shard_headers:
            log_step(f"Fetching safetensors header from '{shard_name}'...")
            req_header_len = urllib.request.Request(file_url, headers={**headers, "Range": "bytes=0-7"})
            with urllib.request.urlopen(req_header_len) as resp:
                header_len = struct.unpack("<Q", resp.read())[0]

            req_header = urllib.request.Request(file_url, headers={**headers, "Range": f"bytes=8-{8 + header_len - 1}"})
            with urllib.request.urlopen(req_header) as resp:
                shard_headers[shard_name] = (json.loads(resp.read().decode("utf-8")), header_len)

        header_json, header_len = shard_headers[shard_name]
        if tensor_key not in header_json:
            raise KeyError(f"Tensor key '{tensor_key}' not found in {shard_name} header")

        meta = header_json[tensor_key]
        shape = meta["shape"]
        offsets = meta["data_offsets"]
        start_byte = 8 + header_len + offsets[0]
        end_byte = 8 + header_len + offsets[1] - 1
        total_bytes = end_byte - start_byte + 1
        size_mb = total_bytes / (1024 * 1024)

        log_step(f"Downloading {tensor_key} ({size_mb:.2f} MB via HTTP Range)...")
        raw_bytes = bytearray()
        curr_offset = start_byte
        chunk_size = 16 * 1024 * 1024
        while curr_offset <= end_byte:
            chunk_end = min(curr_offset + chunk_size - 1, end_byte)
            req_chunk = urllib.request.Request(file_url, headers={**headers, "Range": f"bytes={curr_offset}-{chunk_end}"})
            with urllib.request.urlopen(req_chunk) as resp:
                raw_bytes.extend(resp.read())
            curr_offset = chunk_end + 1

        dtype_str = meta["dtype"]
        dtype_map = {
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "F32": torch.float32,
        }
        pt_dtype = dtype_map.get(dtype_str, torch.float16)
        tensor = torch.frombuffer(raw_bytes, dtype=pt_dtype).reshape(shape).clone().to(dtype=torch.float32)
        cache_path = os.path.join(save_dir, f"layer_{layer_idx}_{name}.pt")
        torch.save(tensor, cache_path)
        layer_weights[name] = tensor

    return layer_weights


def fetch_named_tensor(repo_id, tensor_key, cache_name, save_dir, token=None, revision="main"):
    """
    Fetches a specific named tensor (e.g. embed_tokens, norm, lm_head) via HTTP Range.
    """
    os.makedirs(save_dir, exist_ok=True)
    cache_path = os.path.join(save_dir, f"{cache_name}.pt")
    cached_tensor = load_cached_tensor(cache_path)
    if cached_tensor is not None:
        log_step(f"Loading cached {cache_name} from '{cache_path}'...")
        return cached_tensor

    headers = {"User-Agent": "Mozilla/5.0 (compatible; NoxResearch/1.0)"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    index_url = f"https://huggingface.co/{repo_id}/raw/{revision}/model.safetensors.index.json"
    req_index = urllib.request.Request(index_url, headers=headers)
    with urllib.request.urlopen(req_index) as resp:
        index_json = json.loads(resp.read().decode("utf-8"))
    weight_map = index_json.get("weight_map", {})
    shard_name = weight_map.get(tensor_key, "model-00001-of-00002.safetensors")
    file_url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{shard_name}"

    req_header_len = urllib.request.Request(file_url, headers={**headers, "Range": "bytes=0-7"})
    with urllib.request.urlopen(req_header_len) as resp:
        header_len = struct.unpack("<Q", resp.read())[0]

    req_header = urllib.request.Request(file_url, headers={**headers, "Range": f"bytes=8-{8 + header_len - 1}"})
    with urllib.request.urlopen(req_header) as resp:
        header_json = json.loads(resp.read().decode("utf-8"))

    if tensor_key not in header_json:
        raise KeyError(f"Tensor key '{tensor_key}' not found in {shard_name}")

    meta = header_json[tensor_key]
    shape = meta["shape"]
    offsets = meta["data_offsets"]
    start_byte = 8 + header_len + offsets[0]
    end_byte = 8 + header_len + offsets[1] - 1
    total_bytes = end_byte - start_byte + 1
    size_mb = total_bytes / (1024 * 1024)

    log_step(f"Downloading {tensor_key} ({size_mb:.2f} MB via HTTP Range)...")
    raw_bytes = bytearray()
    curr_offset = start_byte
    chunk_size = 16 * 1024 * 1024
    while curr_offset <= end_byte:
        chunk_end = min(curr_offset + chunk_size - 1, end_byte)
        req_chunk = urllib.request.Request(file_url, headers={**headers, "Range": f"bytes={curr_offset}-{chunk_end}"})
        with urllib.request.urlopen(req_chunk) as resp:
            raw_bytes.extend(resp.read())
        curr_offset = chunk_end + 1

    dtype_str = meta["dtype"]
    dtype_map = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}
    pt_dtype = dtype_map.get(dtype_str, torch.float16)
    tensor = torch.frombuffer(raw_bytes, dtype=pt_dtype).reshape(shape).clone().to(dtype=torch.float32)
    torch.save(tensor, cache_path)
    return tensor


def build_real_token_calibration_data(
    embed_weight, model_id, n_calib=256, n_test=128, seq_len=128, revision="main"
):
    """Build disjoint WikiText train/validation embedding streams with the model tokenizer."""
    from transformers import AutoTokenizer

    train_text, train_meta = load_wikitext2_raw_split("train")
    valid_text, valid_meta = load_wikitext2_raw_split("valid")
    log_step(
        "WikiText-2 raw splits loaded: "
        f"train sha256={train_meta['source_sha256']} ({train_meta['source_bytes']} bytes), "
        f"valid sha256={valid_meta['source_sha256']} ({valid_meta['source_bytes']} bytes)."
    )
    log_step(f"Tokenizing WikiText-2 train/validation with '{model_id}' tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    calib_tokens = tokenizer.encode(train_text, add_special_tokens=True)
    validation_tokens = tokenizer.encode(valid_text, add_special_tokens=True)
    if len(calib_tokens) < n_calib or len(validation_tokens) < n_test:
        raise ValueError(
            "WikiText-2 split is shorter than requested token count: "
            f"train={len(calib_tokens)} requested={n_calib}; "
            f"valid={len(validation_tokens)} requested={n_test}"
        )

    calib_tokens = calib_tokens[:n_calib]
    validation_tokens = validation_tokens[:n_test]
    calib_t = torch.tensor(calib_tokens, dtype=torch.long)
    validation_t = torch.tensor(validation_tokens, dtype=torch.long)

    X_calib = F.embedding(calib_t, embed_weight).T.contiguous()
    X_test = F.embedding(validation_t, embed_weight).T.contiguous()

    def token_digest(token_ids):
        packed = b"".join(int(token_id).to_bytes(8, "little", signed=True) for token_id in token_ids)
        return hashlib.sha256(packed).hexdigest()

    data_metadata = {
        "dataset": "wikitext-2-raw-v1",
        "calibration_split": "train",
        "heldout_split": "validation",
        "evaluation_split": "test",
        "calibration_source": train_meta,
        "heldout_source": valid_meta,
        "tokenizer_model_id": model_id,
        "tokenizer_model_revision": revision,
        "calibration_token_count": len(calib_tokens),
        "heldout_token_count": len(validation_tokens),
        "calibration_token_ids_sha256": token_digest(calib_tokens),
        "heldout_token_ids_sha256": token_digest(validation_tokens),
        "activation_sequence_length": seq_len,
    }
    log_step(
        f"WikiText-2 activations generated: train X_calib {list(X_calib.shape)} "
        f"({len(calib_tokens)} tokens), validation X_test {list(X_test.shape)} "
        f"({len(validation_tokens)} tokens)."
    )
    return X_calib, X_test, data_metadata


def build_calibrated_layer_input(d_model=4096, n_calib=256, n_test=128, seed=42):
    """
    Synthesizes activation inputs with realistic LLM manifold properties.
    """
    torch.manual_seed(seed)
    n_total = n_calib + n_test
    latent_rank = 48
    A = torch.randn(d_model, latent_rank)
    B = torch.randn(latent_rank, n_total)
    X = A @ B

    decay = torch.exp(-torch.linspace(0, 2.5, d_model)).unsqueeze(1)
    X = X * decay

    n_outliers = int(0.02 * d_model)
    outlier_idx = torch.randperm(d_model)[:n_outliers]
    X[outlier_idx, :] *= 18.0

    X = X / (torch.norm(X, dim=0, keepdim=True) + 1e-6) * (d_model ** 0.5)

    X_calib = X[:, :n_calib]
    X_test = X[:, n_calib:]

    return X_calib, X_test


def main():
    parser = argparse.ArgumentParser(description="Day 3 Benchmark: Multi-Layer Sequential Cascade Test")
    parser.add_argument("--start_layer", type=int, default=15, help="Starting layer index (default: 15)")
    parser.add_argument("--end_layer", type=int, default=18, help="Ending layer index inclusive (default: 18)")
    parser.add_argument("--full_model", action="store_true", help="Run full 32 layers (0 to 31)")
    parser.add_argument("--tier", type=str, choices=["extreme_77x", "production_18x", "high_fidelity_10x"], default="extreme_77x", help="Compression tier")
    parser.add_argument("--sweep_cliff", action="store_true", help="Sweep across all 3 tiers to map the Entanglement Cliff")
    parser.add_argument("--save_checkpoint_dir", type=str, default=None, help="Directory to save compressed checkpoints")
    parser.add_argument(
        "--save_no_healing_checkpoint_dir",
        type=str,
        default=None,
        help="Optional output directory for independent Nox checkpoints to evaluate full-model PPL without causal healing.",
    )
    parser.add_argument(
        "--save_validation_activations_dir",
        type=str,
        default=None,
        help="Optional directory for dense WikiText validation activations used by checkpoint output audits.",
    )
    parser.add_argument("--skip_independent", action="store_true", help="Skip compiling Nox Independent baseline to cut compilation runtime in half")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip compiling all baselines (Multiverse + Independent) for 4x faster production compression")
    parser.add_argument("--use_synthetic_calib", action="store_true", help="Use synthetic Gaussian calibration manifold instead of real token embeddings")
    parser.add_argument("--model_id", type=str, default="NousResearch/Llama-2-7b-hf", help="HuggingFace model ID")
    parser.add_argument("--model_revision", type=str, default="main", help="HuggingFace revision (use a commit SHA for repeatable runs)")
    parser.add_argument("--save_dir", type=str, default="./models/Llama-2-7b-hf", help="Local directory for cached weights")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--n_calib", type=int, default=256, help="Number of calibration tokens")
    parser.add_argument("--n_test", type=int, default=128, help="Number of test tokens")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto cuda, else cpu)")
    args = parser.parse_args()

    if args.save_no_healing_checkpoint_dir and (args.skip_independent or args.skip_baselines):
        parser.error("--save_no_healing_checkpoint_dir requires the independent Nox path to be enabled")
    if (
        args.save_checkpoint_dir
        and args.save_no_healing_checkpoint_dir
        and Path(args.save_checkpoint_dir).resolve() == Path(args.save_no_healing_checkpoint_dir).resolve()
    ):
        parser.error("healed and no-healing checkpoints must use different output directories")

    hf_token = args.token or HF_TOKEN
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log_step(f"NVIDIA GPU Acceleration enabled: {torch.cuda.get_device_name(0)} (TF32 active).")
    else:
        log_step(f"Executing on {device.upper()}.")

    if args.full_model:
        start_l = 0
        end_l = 31
    else:
        start_l = args.start_layer
        end_l = args.end_layer

    layer_indices = list(range(start_l, end_l + 1))
    log_step(f"Benchmarking Cascade from Layer {start_l} to Layer {end_l} ({len(layer_indices)} layers total)...")

    def fetcher(l_idx):
        return fetch_layer_weights(
            repo_id=args.model_id,
            layer_idx=l_idx,
            save_dir=args.save_dir,
            token=hf_token,
            revision=args.model_revision,
        )

    # Obtain initial activation stream (real token embeddings by default)
    if args.use_synthetic_calib:
        log_step("Generating synthetic Gaussian calibration manifold...")
        X_calib, X_test = build_calibrated_layer_input(
            d_model=4096,
            n_calib=args.n_calib,
            n_test=args.n_test,
        )
        input_data_metadata = {
            "dataset": "synthetic-gaussian-manifold",
            "calibration_token_count": args.n_calib,
            "heldout_token_count": args.n_test,
            "seed": 42,
            "activation_sequence_length": 128,
        }
    else:
        log_step(f"Loading real token embeddings from '{args.model_id}'...")
        embed_w = fetch_named_tensor(
            repo_id=args.model_id,
            tensor_key="model.embed_tokens.weight",
            cache_name="model_embed_tokens",
            save_dir=args.save_dir,
            token=hf_token,
            revision=args.model_revision,
        )
        X_calib, X_test, input_data_metadata = build_real_token_calibration_data(
            embed_weight=embed_w,
            model_id=args.model_id,
            n_calib=args.n_calib,
            n_test=args.n_test,
            seq_len=128,
            revision=args.model_revision,
        )
        del embed_w
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

        if start_l > 0:
            log_step(f"Propagating real token embeddings through dense Layers 0 -> {start_l - 1}...")
            with torch.no_grad():
                for pre_l in range(0, start_l):
                    pre_weights = fetcher(pre_l)
                    pre_weights = {k: v.to(device=device, dtype=torch.float32) for k, v in pre_weights.items()}
                    X_calib_dev = X_calib.to(device)
                    X_test_dev = X_test.to(device)
                    X_calib_dev, _, _ = forward_transformer_block(pre_weights, X_calib_dev, n_heads=32, n_kv_heads=32, seq_len=128)
                    X_test_dev, _, _ = forward_transformer_block(pre_weights, X_test_dev, n_heads=32, n_kv_heads=32, seq_len=128)
                    X_calib = X_calib_dev.cpu()
                    X_test = X_test_dev.cpu()
                    del pre_weights, X_calib_dev, X_test_dev
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
            log_step(f"True intermediate activations at Layer {start_l} ready.")

    final_norm_w = None
    lm_head_w = None
    if 31 in layer_indices or args.full_model:
        log_step("Fetching final RMSNorm and lm_head weights for Layer 31 exit logit verification...")
        final_norm_w = fetch_named_tensor(
            repo_id=args.model_id,
            tensor_key="model.norm.weight",
            cache_name="model_norm",
            save_dir=args.save_dir,
            token=hf_token,
            revision=args.model_revision,
        )
        lm_head_w = fetch_named_tensor(
            repo_id=args.model_id,
            tensor_key="lm_head.weight",
            cache_name="lm_head",
            save_dir=args.save_dir,
            token=hf_token,
            revision=args.model_revision,
        )

    tiers_to_execute = ["extreme_77x", "production_18x", "high_fidelity_10x"] if args.sweep_cliff else [args.tier]
    sweep_results = []

    for curr_tier in tiers_to_execute:
        tier_save_dir = (
            os.path.join(args.save_checkpoint_dir, curr_tier)
            if args.save_checkpoint_dir and args.sweep_cliff
            else args.save_checkpoint_dir
        )
        independent_save_dir = (
            os.path.join(args.save_no_healing_checkpoint_dir, curr_tier)
            if args.save_no_healing_checkpoint_dir and args.sweep_cliff
            else args.save_no_healing_checkpoint_dir
        )
        if tier_save_dir and os.path.exists(os.path.join(tier_save_dir, "model_manifest.json")):
            manifest_path = os.path.join(tier_save_dir, "model_manifest.json")
            try:
                with open(manifest_path, "r") as mf:
                    manifest_data = json.load(mf)
                identity = {
                    "schema_version": 1,
                    "model_id": args.model_id,
                    "model_revision": args.model_revision,
                    "tier": curr_tier,
                    "config": get_tier_config(curr_tier),
                    "layers": layer_indices,
                    "input_data": input_data_metadata,
                    "sequence_length": 128,
                    "source_fingerprint": source_files_fingerprint(
                        [
                            Path(__file__).resolve(),
                            Path(__file__).resolve().parents[1] / "nox" / "sequential_compiler.py",
                            Path(__file__).resolve().parents[1] / "nox" / "block_compiler.py",
                            Path(__file__).resolve().parents[1] / "nox" / "compiler.py",
                            Path(__file__).resolve().parents[1] / "nox" / "data.py",
                            Path(__file__).resolve().parents[1] / "nox" / "artifacts.py",
                        ]
                    ),
                }
                expected_fingerprint = fingerprint(identity)
                files_ok = (
                    manifest_data.get("checkpoint_file_sizes")
                    == checkpoint_file_sizes(tier_save_dir, layer_indices)
                )
                if (
                    manifest_data.get("run_fingerprint") == expected_fingerprint
                    and manifest_data.get("layers_compressed") == layer_indices
                    and files_ok
                    and not args.save_no_healing_checkpoint_dir
                    and not args.save_validation_activations_dir
                ):
                    log_step(
                        f"Tier '{curr_tier}' matches the current run fingerprint and checkpoint sizes "
                        f"({len(layer_indices)} layers). Skipping verified outputs in {tier_save_dir}."
                    )
                    continue
                log_step(
                    f"Existing outputs for tier '{curr_tier}' are stale or lack verifiable provenance; "
                    "recompiling without deleting them."
                )
            except Exception as e:
                log_step(f"Could not verify existing outputs for '{curr_tier}' ({e}); recompiling in place.")

        identity = {
            "schema_version": 1,
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "tier": curr_tier,
            "config": get_tier_config(curr_tier),
            "layers": layer_indices,
            "input_data": input_data_metadata,
            "sequence_length": 128,
            "source_fingerprint": source_files_fingerprint(
                [
                    Path(__file__).resolve(),
                    Path(__file__).resolve().parents[1] / "nox" / "sequential_compiler.py",
                    Path(__file__).resolve().parents[1] / "nox" / "block_compiler.py",
                    Path(__file__).resolve().parents[1] / "nox" / "compiler.py",
                    Path(__file__).resolve().parents[1] / "nox" / "data.py",
                    Path(__file__).resolve().parents[1] / "nox" / "artifacts.py",
                ]
            ),
        }
        run_fingerprint = fingerprint(identity)

        with torch.no_grad():
            cascade_res = compile_sequential_cascade(
                fetch_layer_fn=fetcher,
                layer_indices=layer_indices,
                X_calib_init=X_calib,
                X_test_init=X_test,
                tier=curr_tier,
                n_heads=32,
                n_kv_heads=32,
                seq_len=128,
                device=device,
                save_checkpoint_dir=tier_save_dir,
                save_independent_checkpoint_dir=independent_save_dir,
                save_validation_activations_dir=args.save_validation_activations_dir,
                skip_independent=args.skip_independent,
                skip_baselines=args.skip_baselines,
                final_norm_weight=final_norm_w,
                lm_head_weight=lm_head_w,
                input_data_metadata=input_data_metadata,
                run_identity=identity,
                run_fingerprint=run_fingerprint,
                verbose=True,
            )

        history = cascade_res["history"]
        cfg = cascade_res["config"]

        print("\nNote: Compile Time is a one-time offline compression cost (0 GPU fine-tuning epochs). Infer Latency is the live forward-pass execution time.")
        print("=" * 140)
        print(f" DAY 3 BENCHMARK: SEQUENTIAL CASCADE SCORECARD (LAYERS {start_l} -> {end_l})")
        print(f" Tier: {cfg['description']}")
        print("=" * 140)
        print(f"{'Layer':<8} | {'MV RelErr':<11} | {'IsoByte RelErr':<16} | {'Indep RelErr':<14} | {'Healed RelErr':<15} | {'Healed CosSim':<14} | {'Compile Time (1-Time)':<21} | {'Infer Latency (ms)':<18}")
        print("-" * 140)

        for h in history:
            l_idx = h["layer_idx"]
            mv_val = h['multiverse']['block']['relative_error']
            mv_err = "N/A (Skip)" if (mv_val != mv_val) else f"{mv_val:.4f}"
            iso_val = h['multiverse_isobyte']['block']['relative_error'] if "multiverse_isobyte" in h else float('nan')
            iso_err = "N/A (Skip)" if (iso_val != iso_val) else f"{iso_val:.4f}"
            indep_val = h['independent']['block']['relative_error']
            indep_err = "N/A (Skip)" if (indep_val != indep_val) else f"{indep_val:.4f}"
            healed_err = f"{h['causal_healed']['block']['relative_error']:.4f}"
            healed_cos = f"{h['causal_healed']['block']['cosine_similarity']:.4f}"
            ctime = f"{h.get('compile_time_sec', 0.0):.2f}s"
            infer = f"{h.get('infer_latency_ms', 0.0):.2f} ms"
            print(f"Layer {l_idx:<2} | {mv_err:<11} | {iso_err:<16} | {indep_err:<14} | {healed_err:<15} | {healed_cos:<14} | {ctime:<21} | {infer:<18}")

        print("-" * 140)

        print("\n" + "=" * 140)
        print(f" BLOCK TRANSFORMATION DELTA (Delta_block) LAYER-BY-LAYER FIDELITY")
        print("=" * 140)
        print(f"{'Layer':<8} | {'MV Delta RelErr':<16} | {'IsoByte Delta RelErr':<21} | {'Indep Delta RelErr':<19} | {'Healed Delta RelErr':<20} | {'MV Delta CosSim':<16} | {'Healed Delta CosSim':<20}")
        print("-" * 140)

        for h in history:
            l_idx = h["layer_idx"]
            mv_d_val = h['multiverse']['delta']['relative_error']
            mv_d_err = "N/A (Skip)" if (mv_d_val != mv_d_val) else f"{mv_d_val:.4f}"
            iso_d_val = h['multiverse_isobyte']['delta']['relative_error'] if "multiverse_isobyte" in h else float('nan')
            iso_d_err = "N/A (Skip)" if (iso_d_val != iso_d_val) else f"{iso_d_val:.4f}"
            indep_d_val = h['independent']['delta']['relative_error']
            indep_d_err = "N/A (Skip)" if (indep_d_val != indep_d_val) else f"{indep_d_val:.4f}"
            healed_d_err = f"{h['causal_healed']['delta']['relative_error']:.4f}"
            mv_d_cos_val = h['multiverse']['delta']['cosine_similarity']
            mv_d_cos = "N/A (Skip)" if (mv_d_cos_val != mv_d_cos_val) else f"{mv_d_cos_val:.4f}"
            healed_d_cos = f"{h['causal_healed']['delta']['cosine_similarity']:.4f}"
            print(f"Layer {l_idx:<2} | {mv_d_err:<16} | {iso_d_err:<21} | {indep_d_err:<19} | {healed_d_err:<20} | {mv_d_cos:<16} | {healed_d_cos:<20}")

        print("-" * 140)

        first_layer_healed_err = history[0]["causal_healed"]["block"]["relative_error"]
        last_layer_healed_err = history[-1]["causal_healed"]["block"]["relative_error"]
        first_layer_indep_err = history[0]["independent"]["block"]["relative_error"]
        last_layer_indep_err = history[-1]["independent"]["block"]["relative_error"]

        healed_drift_growth = (last_layer_healed_err - first_layer_healed_err)
        indep_drift_growth = (last_layer_indep_err - first_layer_indep_err) if (first_layer_indep_err == first_layer_indep_err) else 0.0
        final_delta_cos = history[-1]["causal_healed"]["delta"]["cosine_similarity"]
        avg_layer_kb = sum(h["causal_healed"]["kb"] for h in history) / len(history)
        total_mb = (avg_layer_kb * len(history)) / 1024.0
        serialized_checkpoint_mib = None
        if tier_save_dir:
            serialized_checkpoint_mib = sum(
                checkpoint_file_sizes(tier_save_dir, layer_indices).values()
            ) / (1024.0 * 1024.0)

        sweep_results.append({
            "tier": curr_tier,
            "epsilon": cfg.get("epsilon", "N/A"),
            "desc": cfg["description"],
            "avg_layer_mb": avg_layer_kb / 1024.0,
            "total_mb": total_mb,
            "serialized_checkpoint_mib": serialized_checkpoint_mib,
            "first_err": first_layer_healed_err,
            "last_err": last_layer_healed_err,
            "drift_growth": healed_drift_growth,
            "final_delta_cos": final_delta_cos,
        })

        print("\n[Step 4.2 Inter-Layer Causal Healing Verification]")
        print(f"  First Layer RelErr (Layer {start_l}): Healed = {first_layer_healed_err:.4f} | Independent = {first_layer_indep_err:.4f}")
        print(f"  Last Layer RelErr  (Layer {end_l}): Healed = {last_layer_healed_err:.4f} | Independent = {last_layer_indep_err:.4f}")
        print(f"  Residual Drift Growth across {len(layer_indices)} layers: Healed = {healed_drift_growth:+.4f} | Independent = {indep_drift_growth:+.4f}")
        if abs(healed_drift_growth) <= abs(indep_drift_growth):
            print("  Drift comparison: healing reduced absolute endpoint drift growth versus the independent path.")
        else:
            print("  Drift comparison: healing did not reduce absolute endpoint drift growth versus the independent path.")
        print("  This drift comparison is not an overall model-quality pass; see held-out output and token PPL results.")

    if args.sweep_cliff and len(sweep_results) > 1:
        print("\n" + "=" * 125)
        print(f" DAY 3-4 MASTER ENTANGLEMENT CLIFF & PARETO TRADEOFF SWEEP (LAYERS {start_l} -> {end_l})")
        print("=" * 125)
        print(f"{'Tier':<22} | {'Epsilon':<8} | {'Estimated MB':<13} | {'Serialized MiB':<15} | {'First Err':<10} | {'Final Err':<10} | {'Drift Growth':<13} | {'Delta CosSim':<12}")
        print("-" * 125)
        for r in sweep_results:
            serialized_mib = "N/A" if r["serialized_checkpoint_mib"] is None else f"{r['serialized_checkpoint_mib']:.2f}"
            print(f"{r['tier']:<22} | {str(r['epsilon']):<8} | {r['total_mb']:<13.2f} | {serialized_mib:<15} | {r['first_err']:<10.4f} | {r['last_err']:<10.4f} | {r['drift_growth']:<+13.4f} | {r['final_delta_cos']:<12.4f}")
        print("-" * 125)


if __name__ == "__main__":
    main()
