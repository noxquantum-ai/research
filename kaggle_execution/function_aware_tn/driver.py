#!/usr/bin/env python3
"""Bounded pinned Llama-2 Stage A/B diagnostics for the Kaggle T4 GPU."""

from __future__ import annotations

import gc
import copy
import hashlib
import json
import math
import os
import random
import socket
import shutil
import sys
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
socket.setdefaulttimeout(120)
MODEL_ID = "NousResearch/Llama-2-7b-hf"
MODEL_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TARGET_LAYERS = (0, 15, 31)
SEQUENCE_LENGTH = 128
CALIBRATION_SIZES = (4096, 16384)
HELDOUT_TOKENS = 4096
SPECTRUM_MAX_RANK = 3072
ROOT = Path(__file__).resolve().parent
CODE_MANIFEST = ROOT / "code_manifest.json"
OUT = Path("/kaggle/working/function_aware_tn")
CACHE = Path("/kaggle/working/model_cache")
CHECKPOINTS = OUT / "checkpoints"
RESULTS_PATH = OUT / "results.json"
PROGRESS_PATH = OUT / "progress.json"
sys.path.insert(0, str(ROOT))


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve_secret():
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret("HF_TOKEN")
        return value.strip() if value and value.strip() else None
    except Exception:
        return None


def safe_message(error, token):
    value = f"{type(error).__name__}: {error}"
    return value.replace(token, "[REDACTED]") if token else value


def kaggle_resume_artifact_roots():
    roots = []
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for attached in input_root.iterdir():
            roots.extend((
                attached,
                attached / "function_aware_tn",
                attached / "output" / "function_aware_tn",
            ))
    return [root for root in roots if root.exists()]


def write_blocker(status, detail, token=None):
    record = {
        "status": status,
        "blocker": detail,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": "Salesforce/wikitext/wikitext-2-raw-v1",
        "dataset_revision": DATASET_REVISION,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(RESULTS_PATH, record)
    print(json.dumps(record, indent=2), flush=True)


def ensure_pinned_python_dependencies():
    import importlib.metadata
    from packaging.version import Version

    minimum = {
        "transformers": "4.48.3",
        "huggingface-hub": "0.28.1",
        "pyarrow": "19.0.1",
        "sentencepiece": "0.2.0",
        "tokenizers": "0.21.0",
        "safetensors": "0.5.2",
    }
    current = {}
    missing_or_old = []
    for package, version_floor in minimum.items():
        try:
            current[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            current[package] = None
        if current[package] is None or Version(current[package]) < Version(version_floor):
            missing_or_old.append(package)
    if missing_or_old:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                 "--no-warn-script-location", "-r", str(ROOT / "requirements.txt")],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
        except subprocess.CalledProcessError as exc:
            output = (exc.stdout or "")[-3500:]
            raise RuntimeError(
                f"pinned dependency install failed for {missing_or_old}; pip tail: {output}"
            ) from exc
        current = {
            package: importlib.metadata.version(package) for package in minimum
        }
    return current


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    previous_result = {}
    configured_resume = os.environ.get("NOX_RESUME_DIR")
    previous_result_paths = [RESULTS_PATH]
    if configured_resume:
        configured_root = Path(configured_resume)
        previous_result_paths.extend((
            configured_root / "results.json",
            configured_root / "function_aware_tn" / "results.json",
        ))
    for attached_root in kaggle_resume_artifact_roots():
        previous_result_paths.extend((
            attached_root / "results.json",
            attached_root / "function_aware_tn" / "results.json",
        ))
    for previous_path in previous_result_paths:
        if not previous_path.exists():
            continue
        try:
            candidate = json.loads(previous_path.read_text())
        except Exception:
            continue
        if candidate.get("experiment") == "function_aware_tn_stage_a_stage_b":
            previous_result = candidate
            break
    try:
        software_versions = ensure_pinned_python_dependencies()
    except Exception as exc:
        write_blocker("BLOCKED_DEPENDENCY_INSTALL", safe_message(exc, None))
        return
    token = resolve_secret()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer
    except Exception as exc:
        write_blocker("BLOCKED_DEPENDENCIES", safe_message(exc, token), token)
        return

    if not torch.cuda.is_available():
        write_blocker("BLOCKED_NO_KAGGLE_GPU", "The private kernel has no visible CUDA device.", token)
        return

    device = torch.device("cuda:0")
    torch.manual_seed(20261017)
    random.seed(20261017)
    gauge_permutation = torch.randperm(4096, generator=torch.Generator().manual_seed(20261017))
    gauge_signs = torch.where(
        torch.rand(4096, generator=torch.Generator().manual_seed(20261018)) > 0.5,
        1.0,
        -1.0,
    )
    torch.cuda.reset_peak_memory_stats(device)
    gpu_properties = torch.cuda.get_device_properties(device)
    result = {
        "status": "RUNNING",
        "experiment": "function_aware_tn_stage_a_stage_b",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": "Salesforce/wikitext/wikitext-2-raw-v1",
        "dataset_revision": DATASET_REVISION,
        "target_layers": list(TARGET_LAYERS),
        "calibration_tokens": list(CALIBRATION_SIZES),
        "heldout_tokens": HELDOUT_TOKENS,
        "sequence_length": SEQUENCE_LENGTH,
        "precision": "FP32 module diagnostics; separate FP16 exactness path",
        "software_versions": software_versions,
        "seed": 20261017,
        "code_revision": load_code_revision(),
        "gpu": {
            "name": gpu_properties.name,
            "total_memory_bytes": int(gpu_properties.total_memory),
            "cuda_version": torch.version.cuda,
        },
        "runtime_estimate": {
            "bounded_batch_hours": 6,
            "peak_gpu_memory_budget_gb": 14,
            "persistent_outputs_budget_mb": 100,
            "temporary_resumable_layer_input_checkpoints_max_gb": 1.1,
            "notes": "One streamed 32-layer FP32 forward for 20,480 tokens; 1,024-token sequence chunks; target-layer diagnostics only. Runtime ceiling is the Kaggle CLI's 21,600-second session timeout.",
        },
        "stage_a": {},
        "stage_b": {},
        "post_compression_optimization": False,
        "gradient_updates": False,
        "external_teacher_at_inference": False,
        "dense_residual_sidecars": False,
    }
    atomic_json(RESULTS_PATH, result)

    try:
        from nox.data import load_wikitext2_raw_split
        from benchmarks.benchmark_day3_cascade import (
            fetch_layer_weights,
            fetch_named_tensor,
        )
        from nox.block_compiler import RMSNorm, scaled_dot_product_attention_2d
        from nox.function_aware_benchmarks import (
            benchmark_attention_module,
            benchmark_swiglu_module,
        )
        from nox.representation_symmetries import (
            llama_attention_output,
            rope_commuting_qk_gauge,
            signed_permutation_residual_gauge,
            swiglu_neuron_permutation_gauge,
            swiglu_up_down_scaling_gauge,
            value_output_gauge,
        )
        from nox.spectra import (
            llama_projection_modes,
            singular_spectrum,
            spectra_summary,
            tail_energy_fraction,
            tt_rank_profile,
            tt_site_unfoldings,
        )
        from nox.sequential_compiler import forward_transformer_block
    except Exception as exc:
        write_blocker("BLOCKED_IMPORTS", safe_message(exc, token), token)
        return

    try:
        request = urllib.request.Request(
            f"https://huggingface.co/{MODEL_ID}/raw/{MODEL_REVISION}/model.safetensors.index.json",
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            model_access_status = int(response.status)
    except urllib.error.HTTPError as exc:
        if not token and exc.code in (401, 403):
            write_blocker(
                "BLOCKED_HF_TOKEN_SECRET_MISSING",
                f"Pinned model index returned HTTP {exc.code} without authentication; Kaggle secret HF_TOKEN is required and must have access to this exact revision.",
            )
        else:
            write_blocker(
                "BLOCKED_PINNED_MODEL_ACCESS",
                f"Pinned model index returned HTTP {exc.code}: {safe_message(exc, token)}",
                token,
            )
        return
    except Exception as exc:
        write_blocker("BLOCKED_PINNED_MODEL_ACCESS", safe_message(exc, token), token)
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, token=token,
        )
        train_text, train_source = load_wikitext2_raw_split("train")
        valid_text, valid_source = load_wikitext2_raw_split("validation")
        train_ids = tokenizer.encode(train_text, add_special_tokens=True)
        valid_ids = tokenizer.encode(valid_text, add_special_tokens=True)
        train_windows = sample_windows(train_ids, 16384, 42)
        valid_windows = sample_windows(valid_ids, HELDOUT_TOKENS, 43)
    except Exception as exc:
        write_blocker("BLOCKED_PINNED_MODEL_OR_DATA_ACCESS", safe_message(exc, token), token)
        return

    captured_inputs = {}
    target_weights = {}
    train_token_digest = digest_ids(train_windows)
    valid_token_digest = digest_ids(valid_windows)
    resume_roots = [CHECKPOINTS]
    if configured_resume:
        resume_roots.insert(0, Path(configured_resume))
    resume_roots.extend(
        root / "checkpoints" for root in kaggle_resume_artifact_roots()
    )
    resume_manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "train_token_ids_sha256": train_token_digest,
        "validation_token_ids_sha256": valid_token_digest,
        "sequence_length": SEQUENCE_LENGTH,
        "code_bundle_sha256": load_code_revision().get("bundle_sha256"),
    }
    prior_provenance = previous_result.get("dataset_provenance", {})
    prior_code = previous_result.get("code_revision", {})
    current_code = load_code_revision()
    if (
        previous_result.get("model_id") == MODEL_ID
        and previous_result.get("model_revision") == MODEL_REVISION
        and previous_result.get("dataset_revision") == DATASET_REVISION
        and prior_provenance.get("train_token_ids_sha256") == train_token_digest
        and prior_provenance.get("validation_token_ids_sha256") == valid_token_digest
        and prior_code.get("bundle_sha256") == current_code.get("bundle_sha256")
    ):
        for key in (
            "stage_a", "stage_b", "layer_diagnostic_seconds",
            "real_token_embedding_signed_permutation_relative_error",
            "real_embedding_activation_signed_permutation_relative_error",
        ):
            if key in previous_result:
                result[key] = copy.deepcopy(previous_result[key])
        result["resumed_diagnostic_records"] = {
            "source": "matching results.json with pinned model/data, token hashes, and code bundle digest",
            "stage_a_layers": sorted(result.get("stage_a", {}), key=int),
            "stage_b_layers": sorted(result.get("stage_b", {}), key=int),
        }
        atomic_json(RESULTS_PATH, result)
    resume_layer_indices = []
    for resume_root in resume_roots:
        manifest_path = resume_root / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            saved_manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        if any(saved_manifest.get(key) != value for key, value in resume_manifest.items()):
            continue
        candidate_inputs = {}
        for layer in TARGET_LAYERS:
            path = resume_root / f"layer_{layer}_input.pt"
            if path.exists():
                try:
                    value = torch.load(path, map_location="cpu", weights_only=True)
                except Exception:
                    continue
                if value.ndim == 2 and value.shape == (4096, 20480):
                    candidate_inputs[layer] = value
        usable_layers = []
        for layer in TARGET_LAYERS:
            if layer in candidate_inputs:
                usable_layers.append(layer)
            else:
                break
        if usable_layers and usable_layers[0] == TARGET_LAYERS[0]:
            candidate_inputs = {
                layer: candidate_inputs[layer] for layer in usable_layers
            }
            captured_inputs = candidate_inputs
            resume_layer_indices = sorted(candidate_inputs)
            break

    start_layer = max(resume_layer_indices) if resume_layer_indices else 0
    if resume_layer_indices:
        hidden = captured_inputs[start_layer].to(device=device, dtype=torch.float32)
        probe_token_ids = None
        result["resume_info"] = {
            "resumed": True,
            "resume_root": str(resume_root),
            "start_layer": start_layer,
            "verified_token_manifest": True,
        }
    else:
        try:
            embedding = fetch_named_tensor(
                MODEL_ID,
                "model.embed_tokens.weight",
                "model_embed_tokens",
                str(CACHE),
                token=token,
                revision=MODEL_REVISION,
            )
            token_ids = torch.tensor(train_windows + valid_windows, dtype=torch.long, device=device)
            probe_token_ids = token_ids[:128].detach().cpu()
            embedding_gpu = embedding.to(device=device, dtype=torch.float32)
            hidden = F.embedding(token_ids, embedding_gpu).T.contiguous()
            original_probe = embedding.index_select(0, probe_token_ids)
            transformed_probe = signed_permutation_residual_gauge(
                {"embed_tokens": original_probe}, gauge_permutation, gauge_signs,
            )["embed_tokens"]
            expected_probe = gauge_signs[None, :] * original_probe.index_select(1, gauge_permutation)
            result["real_token_embedding_signed_permutation_relative_error"] = relative_error(
                expected_probe, transformed_probe,
            )
            del embedding, embedding_gpu, token_ids
            gc.collect()
            for item in CACHE.glob("model_embed_tokens.pt"):
                item.unlink(missing_ok=True)
            result["resume_info"] = {"resumed": False, "start_layer": 0}
        except Exception as exc:
            write_blocker("BLOCKED_PINNED_WEIGHT_DOWNLOAD", safe_message(exc, token), token)
            return

    model_forward_started = time.monotonic()
    num_heads, num_kv_heads, d_model = 32, 32, int(hidden.shape[0])
    for layer_index in range(start_layer, 32):
        try:
            weights_cpu = fetch_layer_weights(
                repo_id=MODEL_ID,
                layer_idx=layer_index,
                save_dir=str(CACHE),
                token=token,
                revision=MODEL_REVISION,
            )
        except Exception as exc:
            write_blocker(
                "BLOCKED_PINNED_LAYER_DOWNLOAD",
                f"Layer {layer_index}: {safe_message(exc, token)}",
                token,
            )
            return
        if layer_index in TARGET_LAYERS:
            target_weights[layer_index] = {name: value.clone() for name, value in weights_cpu.items()}
            if layer_index not in captured_inputs:
                captured_inputs[layer_index] = hidden.detach().cpu().clone()
                torch.save(captured_inputs[layer_index], CHECKPOINTS / f"layer_{layer_index}_input.pt")
                atomic_json(CHECKPOINTS / "manifest.json", resume_manifest)
                atomic_json(PROGRESS_PATH, {
                    "status": "RUNNING",
                    "completed_forward_layers": layer_index,
                    "captured_target_layers": sorted(captured_inputs),
                    "checkpoint_files": [str(path.name) for path in CHECKPOINTS.glob("*.pt")],
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
        weights_gpu = {name: value.to(device=device) for name, value in weights_cpu.items()}
        try:
            hidden = stream_forward_block(
                weights_gpu, hidden, forward_transformer_block,
                n_heads=num_heads, n_kv_heads=num_kv_heads,
                seq_len=SEQUENCE_LENGTH, chunk_tokens=1024,
            )
        except Exception as exc:
            result["status"] = "BLOCKED_DENSE_LAYER_FORWARD"
            result["blocker"] = {
                "layer": layer_index,
                "error": safe_message(exc, token),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                "forward_chunk_tokens": 1024,
            }
            atomic_json(RESULTS_PATH, result)
            print(json.dumps(result["blocker"], indent=2), flush=True)
            return
        del weights_gpu, weights_cpu
        for path in CACHE.glob(f"layer_{layer_index}_*.pt"):
            path.unlink(missing_ok=True)
        gc.collect()
        torch.cuda.empty_cache()
    for layer_index in TARGET_LAYERS:
        if layer_index in target_weights:
            continue
        target_weights[layer_index] = fetch_layer_weights(
            repo_id=MODEL_ID,
            layer_idx=layer_index,
            save_dir=str(CACHE),
            token=token,
            revision=MODEL_REVISION,
        )
        for path in CACHE.glob(f"layer_{layer_index}_*.pt"):
            path.unlink(missing_ok=True)
    result["dense_calibration_forward_seconds"] = time.monotonic() - model_forward_started
    result["dataset_provenance"] = {
        "train": train_source,
        "validation": valid_source,
        "train_token_ids_sha256": train_token_digest,
        "validation_token_ids_sha256": valid_token_digest,
        "train_token_count": len(train_windows),
        "validation_token_count": len(valid_windows),
        "sampling": "seeded uniform non-overlapping 128-token windows; sampled window order randomized before fit/selection split",
        "seeds": {"train": 42, "validation": 43},
    }
    result["model_access_probe_http_status"] = model_access_status
    del hidden
    gc.collect()

    for layer_index in TARGET_LAYERS:
        layer_started = time.monotonic()
        weights = {name: value.to(device=device, dtype=torch.float32) for name, value in target_weights[layer_index].items()}
        layer_input = captured_inputs[layer_index].to(device=device, dtype=torch.float32)
        train_x = layer_input[:, :16384]
        heldout_x = layer_input[:, 16384:]
        try:
            attn_input, attn_target, residual1, mlp_input = dense_module_data(
                weights, layer_input, num_heads, num_kv_heads, SEQUENCE_LENGTH,
                RMSNorm, scaled_dot_product_attention_2d, F,
                chunk_tokens=1024,
            )
        except Exception as exc:
            result["status"] = "BLOCKED_TARGET_LAYER_ACTIVATION_EXTRACTION"
            result["blocker"] = {
                "layer": layer_index,
                "error": safe_message(exc, token),
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            atomic_json(RESULTS_PATH, result)
            print(json.dumps(result["blocker"], indent=2), flush=True)
            return
        train_attn_input = attn_input[:, :16384]
        held_attn_input = attn_input[:, 16384:]
        train_attn_target = attn_target[:, :16384]
        held_attn_target = attn_target[:, 16384:]
        train_residual1 = residual1[:, :16384]
        held_residual1 = residual1[:, 16384:]
        train_mlp_input = mlp_input[:, :16384]
        held_mlp_input = mlp_input[:, 16384:]

        try:
            if result.get("stage_a", {}).get(str(layer_index), {}).get("status") == "COMPLETE":
                stage_a_was_resumed = True
            else:
                stage_a_was_resumed = False
                result["stage_a"][str(layer_index)] = measure_stage_a(
                    weights, train_mlp_input.to(device), held_attn_input.to(device),
                    held_mlp_input.to(device),
                    held_x=heldout_x, layer_index=layer_index,
                    num_heads=num_heads, num_kv_heads=num_kv_heads,
                    sequence_length=SEQUENCE_LENGTH, permutation=gauge_permutation,
                    signs=gauge_signs, functions={
                        "llama_attention_output": llama_attention_output,
                        "rope_commuting_qk_gauge": rope_commuting_qk_gauge,
                        "value_output_gauge": value_output_gauge,
                        "swiglu_neuron_permutation_gauge": swiglu_neuron_permutation_gauge,
                        "swiglu_up_down_scaling_gauge": swiglu_up_down_scaling_gauge,
                        "signed_permutation_residual_gauge": signed_permutation_residual_gauge,
                        "forward_transformer_block": forward_transformer_block,
                    },
                    spectrum_functions={
                        "llama_projection_modes": llama_projection_modes,
                        "singular_spectrum": singular_spectrum,
                        "spectra_summary": spectra_summary,
                        "tail_energy_fraction": tail_energy_fraction,
                        "tt_rank_profile": tt_rank_profile,
                        "tt_site_unfoldings": tt_site_unfoldings,
                    },
                    device=device,
                )
            if layer_index == TARGET_LAYERS[0] and not stage_a_was_resumed:
                embedding_input_probe = captured_inputs[0][:, :128].to(
                    device=device, dtype=torch.float32,
                )
                embedding_probe_expected = gauge_signs.to(device)[:, None] * (
                    embedding_input_probe.index_select(0, gauge_permutation.to(device))
                )
                embedding_probe_actual = signed_permutation_residual_gauge(
                    {"embed_tokens": embedding_input_probe.T},
                    gauge_permutation, gauge_signs,
                )["embed_tokens"]
                embedding_error = relative_error(
                    embedding_probe_expected.T, embedding_probe_actual,
                )
                result["real_embedding_activation_signed_permutation_relative_error"] = embedding_error
                stage_a_gate = result["stage_a"][str(layer_index)]["stage_a_gate"]
                stage_a_gate["equivalence_checks"][
                    "real_embedding_activation_relative_error"
                ] = embedding_error
                if embedding_error > 1e-5:
                    stage_a_gate["exact_equivalence_pass"] = False
                    stage_a_gate["provisional_stage_a_pass_for_this_layer"] = False
        except Exception as exc:
            result["stage_a"][str(layer_index)] = {
                "status": "ERROR",
                "error": safe_message(exc, token),
                "traceback": traceback.format_exc().replace(token, "[REDACTED]") if token else traceback.format_exc(),
            }
        atomic_json(RESULTS_PATH, result)

        try:
            prior_stage_b = result.get("stage_b", {}).get(str(layer_index), {})
            if set(prior_stage_b.get("calibration_sizes", {})) == {
                str(value) for value in CALIBRATION_SIZES
            }:
                result["stage_b"][str(layer_index)] = prior_stage_b
            else:
                def save_stage_b_progress(partial):
                    result["stage_b"][str(layer_index)] = copy.deepcopy(partial)
                    result["stage_b_progress"] = {
                        "layer": layer_index,
                        "completed_calibration_sizes": sorted(partial.get("calibration_sizes", {})),
                        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    atomic_json(RESULTS_PATH, result)

                result["stage_b"][str(layer_index)] = run_stage_b_layer(
                    weights=weights,
                    train_attn_input=train_attn_input,
                    train_attn_target=train_attn_target,
                    train_residual1=train_residual1,
                    train_block_input=layer_input[:, :16384],
                    train_mlp_input=train_mlp_input,
                    held_attn_input=held_attn_input,
                    held_attn_target=held_attn_target,
                    held_block_input=heldout_x,
                    held_mlp_input=held_mlp_input,
                    held_residual1=held_residual1,
                    bench_attention=benchmark_attention_module,
                    bench_mlp=benchmark_swiglu_module,
                    downstream_factory=lambda: make_dense_mlp_downstream(weights, RMSNorm, F),
                    sequence_length=SEQUENCE_LENGTH,
                    n_heads=num_heads,
                    n_kv_heads=num_kv_heads,
                    prior_result=prior_stage_b,
                    progress_callback=save_stage_b_progress,
                )
        except Exception as exc:
            result["stage_b"][str(layer_index)] = {
                "status": "ERROR",
                "error": safe_message(exc, token),
                "traceback": traceback.format_exc().replace(token, "[REDACTED]") if token else traceback.format_exc(),
            }
        result["layer_diagnostic_seconds"] = result.get("layer_diagnostic_seconds", {})
        result["layer_diagnostic_seconds"][str(layer_index)] = time.monotonic() - layer_started
        result["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
        atomic_json(RESULTS_PATH, result)
        del weights, layer_input, train_x, heldout_x
        del attn_input, attn_target, residual1, mlp_input
        del train_attn_input, held_attn_input, train_attn_target, held_attn_target
        del train_residual1, held_residual1, train_mlp_input, held_mlp_input
        del captured_inputs[layer_index], target_weights[layer_index]
        gc.collect()
        torch.cuda.empty_cache()

    result["peak_gpu_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    result["global_stage_b_gate"] = evaluate_global_stage_b_gate(result)
    result["status"] = "COMPLETE"
    result["total_runtime_seconds"] = time.monotonic() - model_forward_started
    result["decision"] = stage_decision(result)
    atomic_json(RESULTS_PATH, result)
    shutil.rmtree(CHECKPOINTS, ignore_errors=True)
    atomic_json(PROGRESS_PATH, {
        "status": "COMPLETE",
        "checkpoints_removed_after_success": True,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(json.dumps({
        "status": result["status"],
        "results": str(RESULTS_PATH),
        "peak_gpu_memory_bytes": result["peak_gpu_memory_bytes"],
        "decision": result["decision"],
    }, indent=2), flush=True)


def sample_windows(token_ids, count, seed):
    if count <= 0 or count % SEQUENCE_LENGTH:
        raise ValueError("token count must be a positive multiple of sequence length")
    windows = len(token_ids) // SEQUENCE_LENGTH
    needed = count // SEQUENCE_LENGTH
    if needed > windows:
        raise ValueError(f"split has {windows} complete windows but {needed} were requested")
    chosen = random.Random(seed).sample(range(windows), needed)
    return [token for index in chosen for token in token_ids[
        index * SEQUENCE_LENGTH:(index + 1) * SEQUENCE_LENGTH
    ]]


def digest_ids(token_ids):
    encoded = b"".join(int(value).to_bytes(8, "little", signed=True) for value in token_ids)
    return hashlib.sha256(encoded).hexdigest()


def load_code_revision():
    if CODE_MANIFEST.exists():
        try:
            return json.loads(CODE_MANIFEST.read_text())
        except Exception:
            pass
    return {"bundle_sha256": "unavailable"}


def stream_forward_block(weights, inputs, forward_block, *, n_heads, n_kv_heads, seq_len, chunk_tokens):
    import torch

    if inputs.shape[1] % seq_len or chunk_tokens % seq_len:
        raise ValueError("input and chunk token counts must align to complete sequences")
    outputs = torch.empty_like(inputs)
    with torch.no_grad():
        for start in range(0, inputs.shape[1], int(chunk_tokens)):
            stop = min(inputs.shape[1], start + int(chunk_tokens))
            outputs[:, start:stop] = forward_block(
                weights, inputs[:, start:stop], n_heads=n_heads,
                n_kv_heads=n_kv_heads, seq_len=seq_len, rms_norm_eps=1e-5,
            )[0]
    return outputs


def dense_module_data(
    weights, inputs, n_heads, n_kv_heads, seq_len, RMSNorm, attention, F,
    *, chunk_tokens=1024,
):
    import torch

    d_model = inputs.shape[0]
    output_chunks = [[] for _ in range(4)]
    with torch.no_grad():
        for start in range(0, inputs.shape[1], int(chunk_tokens)):
            stop = min(inputs.shape[1], start + int(chunk_tokens))
            block_input = inputs[:, start:stop]
            norm1 = RMSNorm(d_model, eps=1e-5, weight=weights["input_layernorm"])
            norm2 = RMSNorm(d_model, eps=1e-5, weight=weights["post_attention_layernorm"])
            attention_input = norm1(block_input)
            q = weights["q_proj"] @ attention_input
            k = weights["k_proj"] @ attention_input
            v = weights["v_proj"] @ attention_input
            context = attention(q, k, v, n_heads, n_kv_heads, seq_len)
            attention_output = weights["o_proj"] @ context
            after_attention = block_input + attention_output
            mlp_input = norm2(after_attention)
            for target, value in zip(
                output_chunks,
                (attention_input, attention_output, after_attention, mlp_input),
            ):
                target.append(value.detach().cpu())
    return tuple(torch.cat(chunks, dim=1) for chunks in output_chunks)


def make_dense_mlp_downstream(weights, RMSNorm, F):
    d_model = weights["down_proj"].shape[0]
    norm_weight = weights["post_attention_layernorm"]

    def downstream(after_attention):
        normalized = after_attention * torch.rsqrt(
            after_attention.square().mean(dim=0, keepdim=True) + 1e-5,
        )
        normalized = normalized * norm_weight[:, None]
        output = weights["down_proj"] @ (
            F.silu(weights["gate_proj"] @ normalized) * (weights["up_proj"] @ normalized)
        )
        return after_attention + output

    return downstream


def measure_stage_a(
    weights, calibration_mlp_input, held_attn_input, held_mlp_input, *,
    held_x, layer_index, num_heads,
    num_kv_heads, sequence_length, permutation, signs, functions,
    spectrum_functions, device,
):
    import torch

    started = time.monotonic()
    names = ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj")
    gauge_results = exact_gauge_equivalence(
        weights, calibration_mlp_input, held_attn_input, held_mlp_input, held_x,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        sequence_length=sequence_length, permutation=permutation, signs=signs,
        functions=functions,
    )
    scales = exact_gauge_scales(weights, num_heads, num_kv_heads)
    qk_weights = functions["rope_commuting_qk_gauge"](
        weights, num_heads, num_kv_heads,
        pair_scales=scales["qk_scales"], pair_angles=scales["qk_angles"],
    )
    vo_weights = functions["value_output_gauge"](
        weights, num_heads, num_kv_heads,
        pair_scales=scales["vo_scales"], pair_angles=scales["vo_angles"],
    )
    neuron_importance = (
        torch.nn.functional.silu(weights["gate_proj"] @ calibration_mlp_input)
        * (weights["up_proj"] @ calibration_mlp_input)
    ).square().mean(dim=1)
    neuron_order = torch.argsort(neuron_importance, descending=True, stable=True)
    neuron_weights = functions["swiglu_neuron_permutation_gauge"](weights, neuron_order)
    updown_weights = functions["swiglu_up_down_scaling_gauge"](
        weights, scales["updown_scales"],
    )
    residual_weights = functions["signed_permutation_residual_gauge"](
        weights, permutation, signs,
    )
    representations = {
        "original_nec_identity": (weights, names),
        "rope_commuting_qk_gauge": (qk_weights, ("q_proj", "k_proj")),
        "value_output_headwise_gauge": (vo_weights, ("v_proj", "o_proj")),
        "swiglu_neuron_permutation": (neuron_weights, ("gate_proj", "up_proj", "down_proj")),
        "swiglu_up_down_reciprocal_scaling": (updown_weights, ("up_proj", "down_proj")),
        "residual_signed_permutation": (residual_weights, names),
    }
    result = {
        "status": "COMPLETE",
        "gauge_equivalence": gauge_results,
        "gauge_parameters": {
            "qk_pair_scales": scales["qk_scales"].detach().cpu().tolist(),
            "qk_pair_angles": scales["qk_angles"].detach().cpu().tolist(),
            "vo_pair_scales": scales["vo_scales"].detach().cpu().tolist(),
            "up_down_scales": scales["updown_scales"].detach().cpu().tolist(),
            "neuron_permutation_sha256": hashlib.sha256(
                neuron_order.detach().cpu().numpy().tobytes(),
            ).hexdigest(),
            "residual_permutation_sha256": hashlib.sha256(
                permutation.detach().cpu().numpy().tobytes(),
            ).hexdigest(),
            "deployment_auxiliary_coefficients": 0,
            "folded_into_factor_coordinates": True,
        },
        "operator_spectra": {},
        "tensorization_orders": {
            "nec_original": [0, 1, 2, 3],
            "reverse": [3, 2, 1, 0],
            "interleaved": [0, 2, 1, 3],
        },
    }
    exact_gate = (
        gauge_results["all_fp32_equivalence_pass"]
        and gauge_results["fp16_errors_within_empirical_tolerance"]
    )
    for representation, (candidate_weights, matrix_names) in representations.items():
        for name in matrix_names:
            key = f"{representation}/{name}"
            result["operator_spectra"][key] = measure_operator_spectra(
                candidate_weights[name], spectrum_functions,
                orders=((0, 1, 2, 3), (3, 2, 1, 0), (0, 2, 1, 3))
                if representation == "original_nec_identity" else ((0, 1, 2, 3),),
                seed=20261017 + int(name == "down_proj"),
                repeat_randomized=representation == "original_nec_identity",
            )
    result["stage_a_elapsed_seconds"] = time.monotonic() - started
    result["stage_a_gate"] = evaluate_stage_a_gate(
        result["operator_spectra"], layer_index,
    )
    result["stage_a_gate"]["exact_equivalence_pass"] = exact_gate
    result["stage_a_gate"]["equivalence_checks"] = {
        "fp32_all_transformations_pass": gauge_results["all_fp32_equivalence_pass"],
        "fp16_all_transformations_pass_empirical_tolerance": gauge_results[
            "fp16_errors_within_empirical_tolerance"
        ],
    }
    result["stage_a_gate"]["provisional_stage_a_pass_for_this_layer"] = bool(
        result["stage_a_gate"]["provisional_stage_a_pass_for_this_layer"] and exact_gate
    )
    return result


def exact_gauge_scales(weights, n_heads, n_kv_heads):
    import torch

    d_head = weights["q_proj"].shape[0] // n_heads
    repeat = n_heads // n_kv_heads
    pair_count = d_head // 2
    q = weights["q_proj"].float().reshape(n_heads, d_head, -1)
    k = weights["k_proj"].float().reshape(n_kv_heads, d_head, -1)
    q_first, q_second = q[:, :pair_count], q[:, pair_count:]
    k_first, k_second = k[:, :pair_count], k[:, pair_count:]
    q_energy = (q_first.square() + q_second.square()).sum(dim=2).reshape(
        n_kv_heads, repeat, pair_count,
    ).sum(dim=1)
    k_energy = (k_first.square() + k_second.square()).sum(dim=(0, 2))
    qk_scales = (k_energy.clamp_min(1e-12) / q_energy.clamp_min(1e-12)).pow(0.25).clamp(0.25, 4.0)
    q_pair = torch.stack((q_first, q_second), dim=2).reshape(
        n_kv_heads, repeat, pair_count, 2, q.shape[-1],
    )
    k_pair = torch.stack((k_first, k_second), dim=2)
    q_covariance = torch.einsum("hrpid,hrpjd->hpij", q_pair, q_pair)
    k_covariance = torch.einsum("hpid,hpjd->hpij", k_pair, k_pair)
    qk_covariance = q_covariance + k_covariance
    qk_angles = 0.5 * torch.atan2(
        -2 * qk_covariance[..., 0, 1],
        qk_covariance[..., 0, 0] - qk_covariance[..., 1, 1],
    )

    v = weights["v_proj"].float().reshape(n_kv_heads, d_head, -1)
    o = weights["o_proj"].float().reshape(weights["o_proj"].shape[0], n_kv_heads, repeat, d_head)
    v_pairs = v.reshape(n_kv_heads, pair_count, 2, v.shape[-1])
    v_energy = v_pairs.square().sum(dim=(2, 3))
    o_pairs = o.reshape(o.shape[0], n_kv_heads, repeat, pair_count, 2)
    o_energy = (o_pairs[..., 0].square() + o_pairs[..., 1].square()).sum(dim=(0, 2))
    vo_scales = (o_energy.clamp_min(1e-12) / v_energy.clamp_min(1e-12)).pow(0.25).clamp(0.25, 4.0)
    v_covariance = torch.einsum("hpia,hpja->hpij", v_pairs, v_pairs)
    o_pair = o_pairs.permute(1, 3, 4, 0, 2).reshape(
        n_kv_heads, pair_count, 2, -1,
    )
    o_covariance = torch.einsum("hpia,hpja->hpij", o_pair, o_pair)
    vo_covariance = v_covariance + o_covariance
    vo_angles = 0.5 * torch.atan2(
        -2 * vo_covariance[..., 0, 1],
        vo_covariance[..., 0, 0] - vo_covariance[..., 1, 1],
    )

    up_norm = torch.linalg.vector_norm(weights["up_proj"].float(), dim=1).clamp_min(1e-12)
    down_norm = torch.linalg.vector_norm(weights["down_proj"].float(), dim=0).clamp_min(1e-12)
    updown_scales = torch.sqrt(down_norm / up_norm).clamp(0.25, 4.0)
    return {
        "qk_scales": qk_scales,
        "qk_angles": qk_angles,
        "vo_scales": vo_scales,
        "vo_angles": vo_angles,
        "updown_scales": updown_scales,
    }


def exact_gauge_equivalence(
    weights, mlp_inputs, attn_inputs, heldout_mlp_inputs, block_inputs, *, num_heads,
    num_kv_heads, sequence_length, permutation, signs, functions,
):
    import torch
    import torch.nn.functional as F

    scales = exact_gauge_scales(weights, num_heads, num_kv_heads)
    heldout_mlp = heldout_mlp_inputs[:, -4096:]
    heldout_attn = attn_inputs[:, -4096:]
    heldout_block = block_inputs[:, -4096:]
    mlp_names = ("gate_proj", "up_proj", "down_proj")
    attn_names = ("q_proj", "k_proj", "v_proj", "o_proj")

    def mlp_output(candidate, x):
        return candidate["down_proj"] @ (
            F.silu(candidate["gate_proj"] @ x) * (candidate["up_proj"] @ x)
        )

    scaled = functions["swiglu_up_down_scaling_gauge"](
        weights, scales["updown_scales"],
    )
    permuted = functions["swiglu_neuron_permutation_gauge"](
        weights, torch.argsort(
            (F.silu(weights["gate_proj"] @ mlp_inputs)
             * (weights["up_proj"] @ mlp_inputs)).square().mean(dim=1),
            descending=True, stable=True,
        ),
    )
    mlp_reference = mlp_output(weights, heldout_mlp)
    reference16 = {key: weights[key].half() for key in mlp_names}
    input16 = heldout_mlp.half()
    mlp_reference16 = mlp_output(reference16, input16)
    mlp_fp16_tolerance = max(
        1e-3, 2.0 * relative_error(mlp_reference, mlp_reference16),
    )
    mlp_records = {}
    for name, candidate in (("up_down_reciprocal_scaling", scaled), ("neuron_permutation", permuted)):
        mlp_records[name] = relative_error(mlp_reference, mlp_output(candidate, heldout_mlp))
        candidate16 = {key: value.half() for key, value in candidate.items() if key in mlp_names}
        mlp_records[name + "_fp16"] = relative_error(
            mlp_reference16, mlp_output(candidate16, input16),
        )

    qk = functions["rope_commuting_qk_gauge"](
        weights, num_heads, num_kv_heads,
        pair_scales=scales["qk_scales"], pair_angles=scales["qk_angles"],
    )
    vo = functions["value_output_gauge"](
        weights, num_heads, num_kv_heads,
        pair_scales=scales["vo_scales"], pair_angles=scales["vo_angles"],
    )
    qkvo = functions["value_output_gauge"](
        qk, num_heads, num_kv_heads,
        pair_scales=scales["vo_scales"], pair_angles=scales["vo_angles"],
    )
    attention_records = {}
    def as_sequences(value):
        seq_count = value.shape[1] // sequence_length
        return value[:, :seq_count * sequence_length].T.reshape(
            seq_count, sequence_length, value.shape[0],
        )

    x_attention = as_sequences(heldout_attn)
    reference = functions["llama_attention_output"](
        {name: weights[name] for name in attn_names}, x_attention,
        n_heads=num_heads, n_kv_heads=num_kv_heads,
    )
    reference16 = {key: weights[key].half() for key in attn_names}
    input16 = x_attention.half()
    attention_reference16 = functions["llama_attention_output"](
        reference16, input16, n_heads=num_heads, n_kv_heads=num_kv_heads,
    )
    attention_fp16_tolerance = max(
        1e-3, 2.0 * relative_error(reference, attention_reference16),
    )
    for name, candidate in (("rope_commuting_qk", qk), ("value_output", vo), ("combined_attention_gauges", qkvo)):
        output = functions["llama_attention_output"](
            {key: candidate[key] for key in attn_names}, x_attention,
            n_heads=num_heads, n_kv_heads=num_kv_heads,
        )
        attention_records[name] = relative_error(reference, output)
        candidate16 = {key: candidate[key].half() for key in attn_names}
        transformed16 = functions["llama_attention_output"](
            candidate16, input16, n_heads=num_heads, n_kv_heads=num_kv_heads,
        )
        attention_records[name + "_fp16"] = relative_error(attention_reference16, transformed16)

    residual_gauge = functions["signed_permutation_residual_gauge"](
        weights, permutation, signs,
    )
    x_block = heldout_block
    transformed_x = signs.to(x_block.device)[:, None] * x_block.index_select(0, permutation.to(x_block.device))
    dense_output = functions["forward_transformer_block"](
        weights, x_block, n_heads=num_heads, n_kv_heads=num_kv_heads,
        seq_len=sequence_length, rms_norm_eps=1e-5,
    )[0]
    transformed_output = functions["forward_transformer_block"](
        residual_gauge, transformed_x, n_heads=num_heads, n_kv_heads=num_kv_heads,
        seq_len=sequence_length, rms_norm_eps=1e-5,
    )[0]
    expected = signs.to(dense_output.device)[:, None] * dense_output.index_select(
        0, permutation.to(dense_output.device),
    )
    residual_error = relative_error(expected, transformed_output)
    residual_gauge16 = {key: value.half() for key, value in residual_gauge.items()}
    dense16 = {key: value.half() for key, value in weights.items()}
    x16 = x_block.half()
    x16_new = signs.to(x16.device)[:, None] * x16.index_select(0, permutation.to(x16.device))
    dense_output16 = functions["forward_transformer_block"](
        dense16, x16, n_heads=num_heads, n_kv_heads=num_kv_heads,
        seq_len=sequence_length, rms_norm_eps=1e-5,
    )[0]
    transformed_output16 = functions["forward_transformer_block"](
        residual_gauge16, x16_new, n_heads=num_heads, n_kv_heads=num_kv_heads,
        seq_len=sequence_length, rms_norm_eps=1e-5,
    )[0]
    expected16 = signs.to(dense_output16.device)[:, None] * dense_output16.index_select(
        0, permutation.to(dense_output16.device),
    )
    residual_fp16_tolerance = max(
        1e-3, 2.0 * relative_error(dense_output, dense_output16),
    )
    residual_fp16_error = relative_error(expected16, transformed_output16)
    fp32_errors = list(mlp_records.values()) + list(attention_records.values()) + [residual_error]
    fp32_errors = [value for key, value in zip(
        list(mlp_records.keys()) + list(attention_records.keys()) + ["residual"], fp32_errors,
    ) if not key.endswith("_fp16")]
    return {
        "fp32_relative_error_target": 1e-5,
        "fp16_tolerances": {
            "mlp": mlp_fp16_tolerance,
            "attention": attention_fp16_tolerance,
            "residual_block": residual_fp16_tolerance,
            "rule": "Each path permits max(1e-3, 2x its measured dense FP16-versus-FP32 relative error); 1e-3 is approximately two binary16 unit roundoffs.",
        },
        "dense_fp16_versus_fp32_relative_error": {
            "mlp": relative_error(mlp_reference, mlp_reference16),
            "attention": relative_error(reference, attention_reference16),
            "residual_block": relative_error(dense_output, dense_output16),
        },
        "real_heldout_tokens": int(heldout_block.shape[1]),
        "mlp": mlp_records,
        "attention": attention_records,
        "full_transformer_block_residual_signed_permutation_fp32": residual_error,
        "full_transformer_block_residual_signed_permutation_fp16": residual_fp16_error,
        "full_transformer_block_signed_permutation_fp32_pass": residual_error <= 1e-5,
        "all_fp32_equivalence_pass": all(value <= 1e-5 for value in fp32_errors),
        "fp16_errors_within_empirical_tolerance": all(
            value <= mlp_fp16_tolerance for key, value in mlp_records.items()
            if key.endswith("_fp16")
        ) and all(
            value <= attention_fp16_tolerance for key, value in attention_records.items()
            if key.endswith("_fp16")
        ) and residual_fp16_error <= residual_fp16_tolerance,
    }


def relative_error(reference, candidate):
    import torch

    ref = reference.float()
    cand = candidate.float()
    return float(torch.linalg.vector_norm(ref - cand) / torch.linalg.vector_norm(ref).clamp_min(1e-30))


def measure_operator_spectra(weight, functions, orders, seed, repeat_randomized=False):
    import torch

    modes = functions["llama_projection_modes"](*weight.shape)
    direct = functions["singular_spectrum"](
        weight, exact=False, seed=seed, max_rank=SPECTRUM_MAX_RANK,
        oversample=64, power_iterations=2,
    )
    direct_repeat = None
    if repeat_randomized:
        direct_repeat = functions["singular_spectrum"](
            weight, exact=False, seed=seed + 104729,
            max_rank=SPECTRUM_MAX_RANK, oversample=64, power_iterations=2,
        )
    output = {
        "shape": list(weight.shape),
        "independent_dense_coefficients": int(weight.numel()),
        "direct_operator_spectrum": functions["spectra_summary"](direct),
        "direct_operator_spectrum_method": direct["method"],
        "direct_operator_captured_energy_fraction": direct["captured_energy_fraction"],
        "direct_operator_repeat_seed_capture_delta": (
            abs(direct["captured_energy_fraction"] - direct_repeat["captured_energy_fraction"])
            if direct_repeat is not None else None
        ),
        "tensorizations": {},
    }
    for order in orders:
        cuts, physical = functions["tt_site_unfoldings"](
            weight, modes[0], modes[1], tuple(order),
        )
        cut_records = []
        cut_spectra = []
        for cut_index, cut in enumerate(cuts, start=1):
            spectrum = functions["singular_spectrum"](
                cut, exact=False, seed=seed + cut_index,
                max_rank=SPECTRUM_MAX_RANK, oversample=64, power_iterations=2,
            )
            repeat_delta = None
            if repeat_randomized and tuple(order) == (0, 1, 2, 3) and cut_index == 2:
                repeat = functions["singular_spectrum"](
                    cut, exact=False, seed=seed + 104729,
                    max_rank=SPECTRUM_MAX_RANK, oversample=64, power_iterations=2,
                )
                repeat_delta = abs(
                    spectrum["captured_energy_fraction"] - repeat["captured_energy_fraction"]
                )
                del repeat
            cut_spectra.append(spectrum)
            cut_records.append({
                "cut": cut_index,
                "shape": list(cut.shape),
                "method": spectrum["method"],
                "estimated": spectrum["estimated"],
                "captured_energy_fraction": spectrum["captured_energy_fraction"],
                "independent_seed_capture_delta": repeat_delta,
                "summary": functions["spectra_summary"](spectrum),
            })
        target_records = {}
        for target in (0.01, 0.05, 0.10, 0.20):
            profile = functions["tt_rank_profile"](
                [item["singular_values"] for item in cut_spectra],
                physical,
                math.sqrt(target),
                total_squares=[item["frobenius_norm_squared"] for item in cut_spectra],
            )
            count = int(profile["coefficient_count"])
            svd_rank = count // max(1, sum(weight.shape))
            resolved_svd_rank = min(svd_rank, direct["singular_values"].numel())
            ordinary_error_bound = functions["tail_energy_fraction"](
                direct["singular_values"], resolved_svd_rank,
                direct["frobenius_norm_squared"],
            )
            target_records[str(int(target * 100))] = {
                "requested_relative_frobenius_error": math.sqrt(target),
                "tt_ranks": profile["ranks"],
                "tt_independent_coefficients": count,
                "tt_coefficient_fraction_of_dense": count / weight.numel(),
                "tt_error_estimate_including_unresolved_spectral_energy": profile["tt_error_bound"],
                "estimated_target_met": profile["tt_error_bound"] <= math.sqrt(target),
                "spectrum_complete_for_target": all(
                    record["summary"]["min_rank_by_residual_energy_fraction"][str(target)]["target_reached_with_spectrum"]
                    for record in cut_records
                ),
                "ordinary_svd_rank_at_matched_or_lower_budget": int(svd_rank),
                "ordinary_svd_independent_coefficients": int(svd_rank * sum(weight.shape)),
                "ordinary_svd_residual_energy_at_budget": (
                    ordinary_error_bound if svd_rank <= direct["singular_values"].numel() else None
                ),
                "ordinary_svd_residual_energy_lower_bound_at_spectrum_cap": ordinary_error_bound,
                "ordinary_svd_estimated_relative_frobenius_error": (
                    math.sqrt(ordinary_error_bound) if svd_rank <= direct["singular_values"].numel() else None
                ),
                "ordinary_svd_spectrum_resolution_limited": svd_rank > direct["singular_values"].numel(),
                "spectrum_resolution_limited": any(
                    record["summary"]["min_rank_by_residual_energy_fraction"][str(target)]["target_reached_with_spectrum"] is False
                    for record in cut_records
                ),
            }
        output["tensorizations"]["".join(map(str, order))] = {
            "site_order": list(order),
            "physical_dimensions": list(physical),
            "cuts": cut_records,
            "coefficient_distortion_targets": target_records,
        }
        del cuts, cut_spectra
        torch.cuda.empty_cache()
    return output


def evaluate_stage_a_gate(operator_spectra, layer_index):
    groups = {
        "qk_rope_commutant": ("q_proj", "k_proj"),
        "vo_headwise_rotation": ("v_proj", "o_proj"),
        "swiglu_neuron_permutation": ("gate_proj", "up_proj", "down_proj"),
        "swiglu_up_down_scaling": ("up_proj", "down_proj"),
        "residual_signed_permutation": (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ),
    }
    representation_for_group = {
        "qk_rope_commutant": "rope_commuting_qk_gauge",
        "vo_headwise_rotation": "value_output_headwise_gauge",
        "swiglu_neuron_permutation": "swiglu_neuron_permutation",
        "swiglu_up_down_scaling": "swiglu_up_down_reciprocal_scaling",
        "residual_signed_permutation": "residual_signed_permutation",
    }

    def profile_for(name, representation, site_order="0123"):
        record = operator_spectra.get(f"{representation}/{name}")
        if record is None:
            return None
        tensorization = record["tensorizations"].get(site_order)
        if tensorization is None:
            return None
        return tensorization["coefficient_distortion_targets"]["10"]

    group_results = {}
    for group_name, operator_names in groups.items():
        baseline_count = 0
        transformed_count = 0
        operator_results = {}
        complete = True
        representation = representation_for_group[group_name]
        for operator_name in operator_names:
            baseline_profile = profile_for(
                operator_name, "original_nec_identity", site_order="0123",
            )
            transformed_profile = profile_for(operator_name, representation)
            if (
                baseline_profile is None or transformed_profile is None
                or not baseline_profile["estimated_target_met"]
                or not transformed_profile["estimated_target_met"]
                or not baseline_profile["spectrum_complete_for_target"]
                or not transformed_profile["spectrum_complete_for_target"]
            ):
                operator_results[operator_name] = {"status": "spectrum_or_target_unresolved"}
                complete = False
                continue
            base_count = baseline_profile["tt_independent_coefficients"]
            candidate_count = transformed_profile["tt_independent_coefficients"]
            baseline_count += base_count
            transformed_count += candidate_count
            operator_results[operator_name] = {
                "status": "measured",
                "baseline_coefficients": base_count,
                "transformed_coefficients": candidate_count,
                "coefficient_reduction": 1 - candidate_count / base_count,
            }
        reduction = 1 - transformed_count / baseline_count if baseline_count else None
        group_results[group_name] = {
            "operators": list(operator_names),
            "covered_operators": sum(
                value.get("status") == "measured" for value in operator_results.values()
            ),
            "complete_group_resolved": complete,
            "baseline_coefficients": baseline_count,
            "transformed_coefficients": transformed_count,
            "coefficient_reduction": reduction,
            "meaningful_20pct_reduction": bool(complete and reduction is not None and reduction >= 0.20),
            "operator_details": operator_results,
            "auxiliary_transform_coefficients": 0,
        }

    # Site-order changes are evaluated as a single tensorization-only group;
    # no per-operator mixture with incompatible algebraic gauges is allowed.
    site_order_details = {}
    site_order_baseline = 0
    site_order_candidate = 0
    site_order_complete = True
    for operator_name in (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ):
        baseline_profile = profile_for(operator_name, "original_nec_identity", "0123")
        record = operator_spectra.get(f"original_nec_identity/{operator_name}", {})
        resolved = []
        for order, tensorization in record.get("tensorizations", {}).items():
            candidate_profile = tensorization["coefficient_distortion_targets"]["10"]
            if candidate_profile["estimated_target_met"] and candidate_profile["spectrum_complete_for_target"]:
                resolved.append((candidate_profile["tt_independent_coefficients"], order))
        if (
            baseline_profile is None or not baseline_profile["estimated_target_met"]
            or not baseline_profile["spectrum_complete_for_target"] or not resolved
        ):
            site_order_details[operator_name] = {"status": "spectrum_or_target_unresolved"}
            site_order_complete = False
            continue
        candidate_count, order = min(resolved)
        baseline_count = baseline_profile["tt_independent_coefficients"]
        site_order_baseline += baseline_count
        site_order_candidate += candidate_count
        site_order_details[operator_name] = {
            "status": "measured", "selected_site_order": order,
            "baseline_coefficients": baseline_count,
            "selected_coefficients": candidate_count,
            "coefficient_reduction": 1 - candidate_count / baseline_count,
        }
    site_reduction = (
        1 - site_order_candidate / site_order_baseline if site_order_baseline else None
    )
    site_order_group = {
        "operators": list(site_order_details),
        "covered_operators": sum(
            value.get("status") == "measured" for value in site_order_details.values()
        ),
        "complete_group_resolved": site_order_complete,
        "baseline_coefficients": site_order_baseline,
        "selected_coefficients": site_order_candidate,
        "coefficient_reduction": site_reduction,
        "meaningful_20pct_reduction": bool(
            site_order_complete and site_reduction is not None and site_reduction >= 0.20
        ),
        "auxiliary_transform_coefficients": 0,
        "operator_details": site_order_details,
    }
    group_results["site_order_only"] = site_order_group
    best_group = max(
        group_results,
        key=lambda key: group_results[key]["coefficient_reduction"]
        if group_results[key]["coefficient_reduction"] is not None else -float("inf"),
    )
    passing_groups = [
        name for name, value in group_results.items()
        if value["meaningful_20pct_reduction"]
        and value["covered_operators"] >= (4 if name in ("site_order_only", "residual_signed_permutation") else 2)
    ]
    return {
        "layer_index": int(layer_index),
        "candidate_groups": group_results,
        "best_group": best_group,
        "passing_groups": passing_groups,
        "provisional_stage_a_pass_for_this_layer": bool(passing_groups),
        "coverage_gate": "one coherent exact gauge group must cover its full operator pair/triple; site-order-only evidence needs at least four operators",
        "warning": "Only operators transformed by the same exact gauge are aggregated. Unresolved randomized spectra are excluded; omitted spectral energy is included using exact cut Frobenius norms.",
    }


def run_stage_b_layer(
    *, weights, train_attn_input, train_attn_target, train_residual1,
    train_block_input, train_mlp_input,
    held_attn_input, held_attn_target, held_block_input, held_mlp_input,
    held_residual1, bench_attention, bench_mlp,
    downstream_factory, sequence_length,
    n_heads, n_kv_heads,
    prior_result=None, progress_callback=None,
):
    import torch

    device = weights["q_proj"].device
    result = copy.deepcopy(prior_result) if prior_result else {
        "status": "RUNNING",
        "module_methods": {
            "mlp": ["ordinary_svd", "activation_weighted_svd", "function_aware_down_readout", "MoDeGPT-inspired shared-neuron selection", "previous_grouped_tn"],
            "attention": ["ordinary_svd", "activation_weighted_svd", "function_aware_output_readout", "ordinary_svd_module_output_readout"],
        },
        "candidates_are_training_free": True,
        "post_compression_healing": False,
        "calibration_sizes": {},
    }
    result["status"] = "RUNNING"
    for calibration_size in CALIBRATION_SIZES:
        if str(calibration_size) in result.get("calibration_sizes", {}):
            continue
        calibration_attn = train_attn_input[:, :calibration_size].to(device)
        calibration_attn_target = train_attn_target[:, :calibration_size].to(device)
        calibration_residual = train_residual1[:, :calibration_size].to(device)
        calibration_block_input = train_block_input[:, :calibration_size]
        if calibration_block_input.device != device:
            calibration_block_input = calibration_block_input.to(device)
        calibration_mlp = train_mlp_input[:, :calibration_size].to(device)
        fit_count = (calibration_size * 3 // 4 // SEQUENCE_LENGTH) * SEQUENCE_LENGTH
        fit_slice = slice(0, fit_count)
        selection_slice = slice(fit_count, calibration_size)
        fit_attn = calibration_attn[:, fit_slice]
        selection_attn = calibration_attn[:, selection_slice]
        fit_attn_target = calibration_attn_target[:, fit_slice]
        selection_attn_target = calibration_attn_target[:, selection_slice]
        fit_residual = calibration_residual[:, fit_slice]
        selection_residual = calibration_residual[:, selection_slice]
        fit_block_residual = calibration_block_input[:, fit_slice]
        selection_block_residual = calibration_block_input[:, selection_slice]
        fit_mlp = calibration_mlp[:, fit_slice]
        selection_mlp = calibration_mlp[:, selection_slice]
        heldout_attn_residual = held_block_input.to(device)
        heldout_mlp_residual = held_residual1.to(device)
        held_attn_input_device = held_attn_input.to(device)
        held_attn_target_device = held_attn_target.to(device)
        held_mlp_input_device = held_mlp_input.to(device)
        downstream = downstream_factory()
        result["calibration_sizes"][str(calibration_size)] = {
            "fit_tokens": fit_count,
            "internal_selection_tokens": calibration_size - fit_count,
            "heldout_tokens": HELDOUT_TOKENS,
            "statistics_method": "MLP and attention input covariances plus MLP hidden covariance are accumulated in 512-token chunks; the function-aware Down fit uses a seeded Gaussian token sketch built from all fit tokens.",
            "65k_calibration_status": "omitted from this first bounded batch; the requested 4,096- and 16,384-token calibration sizes are measured",
            "mlp": bench_mlp(
                {name: weights[name] for name in ("gate_proj", "up_proj", "down_proj")},
                fit_mlp, selection_mlp, held_mlp_input_device,
                fit_residual=fit_residual,
                selection_residual=selection_residual,
                heldout_residual=heldout_mlp_residual,
                budgets=(0.75, 0.50, 0.25),
                include_grouped=True,
                seed=20261017,
            ),
            "attention": bench_attention(
                {name: weights[name] for name in ("q_proj", "k_proj", "v_proj", "o_proj")},
                fit_attn,
                fit_attn_target,
                selection_attn,
                selection_attn_target,
                held_attn_input_device,
                held_attn_target_device,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                seq_len=sequence_length,
                fit_residual=fit_block_residual,
                selection_residual=selection_block_residual,
                heldout_residual=heldout_attn_residual,
                downstream_fn=downstream,
                budgets=(0.75, 0.50, 0.25),
                seed=20261017,
            ),
        }
        if progress_callback is not None:
            progress_callback(result)
        del downstream
        del calibration_attn, calibration_attn_target, calibration_residual
        del calibration_block_input, calibration_mlp
        del fit_attn, selection_attn, fit_attn_target, selection_attn_target
        del fit_residual, selection_residual, fit_block_residual, selection_block_residual
        del fit_mlp, selection_mlp
        del heldout_attn_residual, heldout_mlp_residual
        del held_attn_input_device, held_attn_target_device, held_mlp_input_device
        gc.collect()
        torch.cuda.empty_cache()
    result["status"] = "COMPLETE"
    return result


def evaluate_global_stage_b_gate(result):
    import torch

    entries = []
    for module_name, output_key in (
        ("mlp", "mlp_output"),
        ("attention", "attention_output"),
    ):
        for calibration_size in CALIBRATION_SIZES:
            for fraction in ("0.75", "0.5", "0.25"):
                layer_errors = []
                for layer in TARGET_LAYERS:
                    layer_record = result.get("stage_b", {}).get(str(layer), {})
                    size_record = layer_record.get("calibration_sizes", {}).get(str(calibration_size), {})
                    budget = size_record.get(module_name, {}).get("budgets", {}).get(fraction)
                    if budget is None:
                        layer_errors = []
                        break
                    candidates = budget["candidates"]
                    candidate_error = candidates["function_aware"]["metrics"]["heldout"][output_key]["relative_error"]
                    baseline_errors = {
                        name: entry["metrics"]["heldout"][output_key]["relative_error"]
                        for name, entry in candidates.items() if name != "function_aware"
                    }
                    best_name = min(baseline_errors, key=baseline_errors.get)
                    best_error = baseline_errors[best_name]
                    layer_errors.append({
                        "layer": layer,
                        "candidate_error": candidate_error,
                        "best_baseline": best_name,
                        "best_baseline_error": best_error,
                        "relative_improvement": (best_error - candidate_error) / max(best_error, 1e-30),
                    })
                median_improvement = (
                    float(torch.tensor([item["relative_improvement"] for item in layer_errors]).median())
                    if layer_errors else None
                )
                min_improvement = min(
                    (item["relative_improvement"] for item in layer_errors),
                    default=None,
                )
                entries.append({
                    "module": module_name,
                    "calibration_tokens": calibration_size,
                    "coefficient_fraction": fraction,
                    "per_layer": layer_errors,
                    "median_relative_improvement": median_improvement,
                    "minimum_layer_relative_improvement": min_improvement,
                    "passes_15pct_median_gate": bool(
                        len(layer_errors) == len(TARGET_LAYERS)
                        and median_improvement >= 0.15 and min_improvement >= -0.10
                    ),
                })
    passed = any(item["passes_15pct_median_gate"] for item in entries)
    return {
        "provisional_improvement_gate": "at least 15% lower median heldout error than the best matched baseline across layers 0, 15, 31, with no layer more than 10% worse",
        "points_passing": sum(item["passes_15pct_median_gate"] for item in entries),
        "points_evaluated": len(entries),
        "details": entries,
        "provisional_stage_b_pass": passed,
    }


def stage_decision(result):
    stage_a_records = [
        record for record in result.get("stage_a", {}).values()
        if record.get("status") == "COMPLETE"
    ]
    layer_passes = sum(
        record.get("stage_a_gate", {}).get("provisional_stage_a_pass_for_this_layer", False)
        for record in stage_a_records
    )
    stage_a_pass = len(stage_a_records) == len(TARGET_LAYERS) and layer_passes >= 2
    stage_b_pass = result.get("global_stage_b_gate", {}).get("provisional_stage_b_pass", False)
    return {
        "stage_a": "GO" if stage_a_pass else "NO-GO or incomplete",
        "stage_b": "GO" if stage_b_pass else "NO-GO or incomplete",
        "candidate_2": "supported only if Stage A passes; otherwise unsupported",
        "full_model_compilation": "not run in this batch; requires a clear module/prefix GO",
    }


if __name__ == "__main__":
    main()
