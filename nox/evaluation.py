"""Fail-closed validation helpers for full-model NEC evaluations."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from nox.artifacts import fingerprint
from nox.compiler import reconstruct_from_artifact


MODEL_ID = "NousResearch/Llama-2-7b-hf"
MODEL_REVISION = "8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
PROJECTION_SHAPES = {
    "q_proj": (4096, 4096),
    "k_proj": (4096, 4096),
    "v_proj": (4096, 4096),
    "o_proj": (4096, 4096),
    "gate_proj": (11008, 4096),
    "up_proj": (11008, 4096),
    "down_proj": (4096, 11008),
}
DENSE_LLAMA2_7B_COEFFICIENTS = 6_738_415_616
UNCHANGED_COEFFICIENTS = 2 * 32_000 * 4096 + 4096
EXPECTED_LLAMA2_CONFIG = {
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "vocab_size": 32000,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 0,
}
TOKENIZER_PROBE_TEXT = "NEC tokenizer compatibility probe."
TOKENIZER_PROBE_IDS = [1, 405, 11206, 5993, 3950, 24521, 410, 915, 29889]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} is empty or contains non-finite values")


def validate_llama2_config(config) -> dict:
    """Fail closed on incompatible model dimensions or normalization settings."""
    actual_config = {key: getattr(config, key, None) for key in EXPECTED_LLAMA2_CONFIG}
    for key, expected_value in EXPECTED_LLAMA2_CONFIG.items():
        actual = actual_config[key]
        if isinstance(expected_value, float):
            matches = isinstance(actual, (int, float)) and math.isclose(
                float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12,
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise ValueError(
                f"Pinned Llama-2 config mismatch for {key}: {actual!r}; "
                f"expected {expected_value!r}"
            )
    architectures = list(getattr(config, "architectures", []) or [])
    if architectures != ["LlamaForCausalLM"]:
        raise ValueError(f"Expected LlamaForCausalLM architecture, found {architectures}")
    return {
        "model_config": actual_config,
        "architectures": architectures,
        "model_config_commit_hash": getattr(config, "_commit_hash", None),
    }


def validate_llama2_model_tokenizer(model, tokenizer) -> dict:
    """Fail closed on Llama-2 architecture or tokenizer incompatibilities."""
    config = model.config
    config_audit = validate_llama2_config(config)
    tokenizer_vocab = int(getattr(tokenizer, "vocab_size", -1))
    tokenizer_length = len(tokenizer)
    if tokenizer_vocab != config.vocab_size or tokenizer_length != config.vocab_size:
        raise ValueError(
            f"Tokenizer/model vocabulary mismatch: tokenizer vocab={tokenizer_vocab}, "
            f"tokenizer length={tokenizer_length}, model vocab={config.vocab_size}"
        )
    special_ids = {
        "bos": getattr(tokenizer, "bos_token_id", None),
        "eos": getattr(tokenizer, "eos_token_id", None),
        "unk": getattr(tokenizer, "unk_token_id", None),
        "pad": getattr(tokenizer, "pad_token_id", None),
    }
    expected_special_ids = {"bos": 1, "eos": 2, "unk": 0, "pad": 0}
    if special_ids != expected_special_ids:
        raise ValueError(
            f"Tokenizer special-token IDs mismatch: {special_ids}; "
            f"expected {expected_special_ids}"
        )
    probe = tokenizer(TOKENIZER_PROBE_TEXT, add_special_tokens=True)
    probe_ids = list(probe["input_ids"])
    if [int(token_id) for token_id in probe_ids] != TOKENIZER_PROBE_IDS:
        raise ValueError(
            f"Pinned tokenizer probe tokenization mismatch: {probe_ids}; "
            f"expected {TOKENIZER_PROBE_IDS}"
        )
    if any(int(token_id) < 0 or int(token_id) >= config.vocab_size for token_id in probe_ids):
        raise ValueError("Tokenizer emitted an ID outside the model vocabulary")
    embedding = model.get_input_embeddings()
    output = model.get_output_embeddings()
    if embedding.weight.shape[0] != config.vocab_size or output.weight.shape[0] != config.vocab_size:
        raise ValueError("Model embedding or language-head dimensions disagree with vocabulary")
    layer_count = len(model.model.layers)
    if layer_count != config.num_hidden_layers:
        raise ValueError("Loaded decoder layer count disagrees with model configuration")
    for layer_index, layer in enumerate(model.model.layers):
        modules = {
            "q_proj": layer.self_attn.q_proj,
            "k_proj": layer.self_attn.k_proj,
            "v_proj": layer.self_attn.v_proj,
            "o_proj": layer.self_attn.o_proj,
            "gate_proj": layer.mlp.gate_proj,
            "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
        }
        for name, module in modules.items():
            expected_shape = PROJECTION_SHAPES[name]
            if tuple(module.weight.shape) != expected_shape:
                raise ValueError(
                    f"Layer {layer_index} {name} has shape {tuple(module.weight.shape)}, "
                    f"expected {expected_shape}"
                )
    return {
        **config_audit,
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "name_or_path": getattr(tokenizer, "name_or_path", None),
            "vocab_size": tokenizer_vocab,
            "length": tokenizer_length,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "special_token_ids": special_ids,
        "probe_text": TOKENIZER_PROBE_TEXT,
        "probe_token_ids": [int(token_id) for token_id in probe_ids],
        "probe_token_ids_sha256": hashlib.sha256(
            json.dumps([int(token_id) for token_id in probe_ids], separators=(",", ":")).encode()
        ).hexdigest(),
            "add_bos_token": getattr(tokenizer, "add_bos_token", None),
            "add_eos_token": getattr(tokenizer, "add_eos_token", None),
            "chat_template_present": getattr(tokenizer, "chat_template", None) is not None,
            "special_tokens_map": {
                key: str(value)
                for key, value in getattr(tokenizer, "special_tokens_map", {}).items()
            },
            "pinned_revision": MODEL_REVISION,
        },
        "projection_layers": layer_count,
    }


def validate_mpo_artifact(
    name: str,
    artifact: dict,
    expected_shape: tuple[int, int],
    expected_ranks: tuple[int, ...] | list[int] | None = None,
) -> dict:
    """Validate one serialized four-site NEC projection without trusting metadata."""
    if artifact.get("codec_type") not in (None, "nec_mpo_v1"):
        raise ValueError(f"{name}: expected nec_mpo_v1 artifact")
    if tuple(artifact.get("orig_shape", ())) != expected_shape:
        raise ValueError(f"{name}: projection shape mismatch")
    cores = artifact.get("cores")
    if not isinstance(cores, list) or len(cores) != 4:
        raise ValueError(f"{name}: expected four MPO cores")
    previous_rank = 1
    output_product = input_product = 1
    coefficient_count = 0
    core_shapes = []
    for index, core in enumerate(cores):
        if not torch.is_tensor(core) or core.ndim != 4:
            raise ValueError(f"{name}: malformed MPO core {index}")
        if core.dtype != torch.float16:
            raise ValueError(f"{name}: expected serialized FP16 core {index}")
        _finite_tensor(f"{name}.cores[{index}]", core)
        left, output_mode, input_mode, right = map(int, core.shape)
        if left != previous_rank or output_mode < 1 or input_mode < 1 or right < 1:
            raise ValueError(f"{name}: incompatible MPO bond at core {index}")
        previous_rank = right
        output_product *= output_mode
        input_product *= input_mode
        coefficient_count += core.numel()
        core_shapes.append(list(core.shape))
    if previous_rank != 1 or (output_product, input_product) != expected_shape:
        raise ValueError(f"{name}: MPO physical dimensions do not reconstruct the projection")
    actual_ranks = [1, *(shape[3] for shape in core_shapes[:-1]), 1]
    stored_ranks = artifact.get("ranks")
    if stored_ranks is not None and list(stored_ranks) != actual_ranks:
        raise ValueError(f"{name}: stored ranks disagree with core tensor shapes")
    if expected_ranks is not None and actual_ranks != list(expected_ranks):
        raise ValueError(
            f"{name}: expected High-Fidelity bond ranks {list(expected_ranks)}, "
            f"found {actual_ranks}"
        )

    scale = artifact.get("d_awt")
    if not torch.is_tensor(scale) or tuple(scale.shape) != (expected_shape[1],):
        raise ValueError(f"{name}: malformed activation scale")
    if scale.dtype != torch.float16:
        raise ValueError(f"{name}: activation scale must be serialized as FP16")
    _finite_tensor(f"{name}.d_awt", scale)
    if not torch.all(scale > 0):
        raise ValueError(f"{name}: activation scale must be strictly positive")
    coefficient_count += scale.numel()

    left, right = artifact.get("A_bridge"), artifact.get("B_bridge")
    if (left is None) != (right is None):
        raise ValueError(f"{name}: bridge factors must occur together")
    bridge_rank = 0
    if left is not None:
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            raise ValueError(f"{name}: malformed bridge factors")
        if left.dtype != torch.float16 or right.dtype != torch.float16:
            raise ValueError(f"{name}: bridge factors must be serialized as FP16")
        if left.ndim != 2 or right.ndim != 2 or left.shape[0] != expected_shape[0]:
            raise ValueError(f"{name}: bridge output shape mismatch")
        if right.shape[1] != expected_shape[1] or left.shape[1] != right.shape[0]:
            raise ValueError(f"{name}: bridge bond mismatch")
        _finite_tensor(f"{name}.A_bridge", left)
        _finite_tensor(f"{name}.B_bridge", right)
        bridge_rank = int(left.shape[1])
        coefficient_count += left.numel() + right.numel()

    indices, values = artifact.get("sparse_indices"), artifact.get("sparse_values")
    if (indices is None) != (values is None):
        raise ValueError(f"{name}: sparse indices and values must occur together")
    sparse_count = 0
    if indices is not None:
        if not torch.is_tensor(indices) or not torch.is_tensor(values):
            raise ValueError(f"{name}: malformed sparse tensors")
        if indices.dtype != torch.int32 or values.dtype != torch.float16:
            raise ValueError(f"{name}: expected int32 coordinates and FP16 sparse values")
        if indices.ndim != 2 or indices.shape[1] != 2 or values.ndim != 1:
            raise ValueError(f"{name}: malformed sparse tensor dimensions")
        if indices.shape[0] != values.shape[0]:
            raise ValueError(f"{name}: sparse index/value count mismatch")
        _finite_tensor(f"{name}.sparse_values", values)
        rows, columns = indices[:, 0].long(), indices[:, 1].long()
        if (rows < 0).any() or (rows >= expected_shape[0]).any():
            raise ValueError(f"{name}: sparse row out of range")
        if (columns < 0).any() or (columns >= expected_shape[1]).any():
            raise ValueError(f"{name}: sparse column out of range")
        linear = rows * expected_shape[1] + columns
        if torch.unique(linear).numel() != linear.numel():
            raise ValueError(f"{name}: duplicate sparse coordinates")
        sparse_count = int(values.numel())
        coefficient_count += sparse_count

    return {
        "shape": list(expected_shape),
        "core_shapes": core_shapes,
        "bridge_rank": bridge_rank,
        "sparse_coefficients": sparse_count,
        "independent_numerical_coefficients": int(coefficient_count),
    }


def validate_high_fidelity_checkpoint(
    checkpoint_dir: str | Path,
    *,
    reconstruct: bool = False,
    reconstruction_device: str = "cpu",
    expected_calibration_tokens: int = 256,
    expected_heldout_tokens: int = 128,
    allow_partial_prefix: bool = False,
    expected_quantization_schedule: str = "in_sweep",
    expected_cg_threshold: int = 4096,
    expected_cg_iterations: int = 6,
) -> dict:
    """Audit a full checkpoint or an explicitly permitted contiguous prefix.

    Defaults preserve the historical 256/128-token checkpoint contract. New
    RunPod experiments must pass their configured token counts explicitly.
    A partial checkpoint is accepted only when ``allow_partial_prefix`` is set
    and its compressed layers are exactly ``0, ..., k-1``.
    """
    directory = Path(checkpoint_dir)
    manifest_path = directory / "model_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.get("run_identity", {})
    if identity.get("model_id") != MODEL_ID or identity.get("model_revision") != MODEL_REVISION:
        raise ValueError("Checkpoint model identity mismatch")
    if identity.get("compression_path") != "no_healing_independent":
        raise ValueError("Checkpoint is not the current no-healing independent NEC path")
    config = identity.get("config", {})
    expected = {
        "ranks": [1, 64, 128, 64, 1],
        "r_bridge": 128,
        "n_outliers": 524288,
        "mpo_n_sweeps": 2,
        "mpo_alpha_awt": 0.5,
    }
    actual = {
        "ranks": list(config.get("ranks", [])),
        "r_bridge": config.get("r_bridge"),
        "n_outliers": config.get("n_outliers"),
        "mpo_n_sweeps": identity.get("mpo_n_sweeps", config.get("mpo_n_sweeps", 2)),
        "mpo_alpha_awt": identity.get("mpo_alpha_awt", config.get("mpo_alpha_awt", 0.5)),
    }
    if actual != expected:
        raise ValueError(f"Checkpoint is not the exact High-Fidelity configuration: {actual}")
    if identity.get("calibration_activation_mode", "dense") != "dense":
        raise ValueError("Checkpoint did not use independent dense calibration activations")
    if identity.get("construction_mode", "independent") != "independent":
        raise ValueError("Checkpoint construction mode mismatch")
    layers_declared = identity.get("layers")
    if not isinstance(layers_declared, list) or not layers_declared:
        raise ValueError("Manifest must declare at least one compressed layer")
    if layers_declared != list(range(len(layers_declared))):
        raise ValueError("Compressed layer declaration must be a contiguous prefix starting at layer 0")
    if len(layers_declared) != 32 and not allow_partial_prefix:
        raise ValueError("Manifest does not declare all 32 layers; pass allow_partial_prefix explicitly")
    if len(layers_declared) > 32:
        raise ValueError("Manifest declares more layers than Llama-2-7B contains")
    if manifest.get("layers_compressed") != layers_declared:
        raise ValueError("Manifest compressed-layer list disagrees with run identity")
    actual_schedule = identity.get("mpo_quantization_schedule", "in_sweep")
    actual_cg_threshold = identity.get("mpo_cg_threshold", 4096)
    actual_cg_iterations = identity.get("mpo_cg_iterations", 6)
    if (actual_schedule, actual_cg_threshold, actual_cg_iterations) != (
        expected_quantization_schedule, expected_cg_threshold, expected_cg_iterations,
    ):
        raise ValueError(
            "MPO solver/quantization identity mismatch: "
            f"got {(actual_schedule, actual_cg_threshold, actual_cg_iterations)}, expected "
            f"{(expected_quantization_schedule, expected_cg_threshold, expected_cg_iterations)}"
        )
    forward = identity.get("forward_options", {})
    if forward != {"n_heads": 32, "n_kv_heads": 32, "seq_len": 128, "rms_norm_eps": 1e-5}:
        raise ValueError(f"Llama-2 forward configuration mismatch: {forward}")
    inputs = identity.get("input_data", {})
    if (
        not isinstance(expected_calibration_tokens, int)
        or not isinstance(expected_heldout_tokens, int)
        or expected_calibration_tokens <= 0
        or expected_heldout_tokens <= 0
        or expected_calibration_tokens % forward["seq_len"]
        or expected_heldout_tokens % forward["seq_len"]
    ):
        raise ValueError("Expected calibration counts must be positive multiples of seq_len")
    if (
        inputs.get("calibration_token_count") != expected_calibration_tokens
        or inputs.get("heldout_token_count") != expected_heldout_tokens
    ):
        raise ValueError(
            "Calibration token counts differ from the requested protocol: "
            f"got {inputs.get('calibration_token_count')}/{inputs.get('heldout_token_count')}, "
            f"expected {expected_calibration_tokens}/{expected_heldout_tokens}"
        )
    for key in ("calibration_source", "heldout_source"):
        if inputs.get(key, {}).get("dataset_revision") != WIKITEXT_REVISION:
            raise ValueError(f"Unpinned calibration provenance in {key}")
    if fingerprint(identity) != manifest.get("run_fingerprint"):
        raise ValueError("Checkpoint manifest fingerprint mismatch")
    accepted_artifact_fingerprints = {manifest["run_fingerprint"]}
    for event in manifest.get("resume_events", []):
        previous_fingerprint = event.get("previous_run_fingerprint")
        if (
            isinstance(previous_fingerprint, str)
            and len(previous_fingerprint) == 64
            and all(character in "0123456789abcdef" for character in previous_fingerprint)
        ):
            accepted_artifact_fingerprints.add(previous_fingerprint)

    compressed_layers = list(layers_declared)
    expected_files = {f"compressed_layer_{index}.pt" for index in compressed_layers}
    actual_files = {path.name for path in directory.glob("compressed_layer_*.pt")}
    if actual_files != expected_files:
        raise ValueError(
            "Checkpoint directory layer files do not match its declared prefix: "
            f"found {sorted(actual_files)}, expected {sorted(expected_files)}"
        )
    if list(directory.glob("*.pt.partial")):
        raise ValueError("Incomplete temporary layer checkpoint present")
    file_sizes = manifest.get("checkpoint_file_sizes", {})
    file_hashes = manifest.get("checkpoint_file_sha256", {})
    if set(file_sizes) != expected_files or set(file_hashes) != expected_files:
        raise ValueError("Manifest size/hash entries must match all declared layer files")
    if sum(int(size) for size in file_sizes.values()) != int(
        manifest.get("serialized_checkpoint_bytes", -1)
    ):
        raise ValueError("Manifest serialized checkpoint byte total is inconsistent")
    expected_intermediate = {
        f"{prefix}_layer_{layer}.pt"
        for layer in compressed_layers
        for prefix in (
            "dense_input",
            "calibration_input",
            "layer_input_validation",
        )
    }
    intermediate = manifest.get("intermediate_files", {})
    if set(intermediate) != expected_intermediate:
        raise ValueError("Manifest intermediate metadata must match every declared layer")
    actual_intermediate = set()
    for prefix in ("dense_input", "calibration_input", "layer_input_validation"):
        actual_intermediate.update(path.name for path in directory.glob(f"{prefix}_layer_*.pt"))
    if actual_intermediate != expected_intermediate:
        raise ValueError("Checkpoint directory is missing a saved layer input artifact")
    intermediate_bytes = 0
    for filename, metadata in intermediate.items():
        path = directory / filename
        size = path.stat().st_size
        if size != int(metadata.get("bytes", -1)) or sha256_file(path) != metadata.get("sha256"):
            raise ValueError(f"Intermediate artifact integrity failure: {filename}")
        intermediate_bytes += size

    layers = []
    projection_coefficients = 0
    layer_norm_coefficients = 0
    serialized_bytes = 0
    for index in compressed_layers:
        path = directory / f"compressed_layer_{index}.pt"
        file_sha256 = sha256_file(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("layer_idx") != index:
            raise ValueError(f"Layer index mismatch in {path.name}")
        if checkpoint.get("run_fingerprint") not in accepted_artifact_fingerprints:
            raise ValueError(f"Run fingerprint mismatch in {path.name}")
        artifacts = checkpoint.get("compressed_artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(PROJECTION_SHAPES):
            raise ValueError(f"Projection set mismatch in {path.name}")
        projection_reports = {}
        for projection, shape in PROJECTION_SHAPES.items():
            artifact = artifacts[projection]
            projection_report = validate_mpo_artifact(
                f"layer{index}.{projection}",
                artifact,
                shape,
                expected_ranks=expected["ranks"],
            )
            if projection_report["bridge_rank"] != expected["r_bridge"]:
                raise ValueError(
                    f"layer{index}.{projection}: bridge rank mismatch "
                    f"({projection_report['bridge_rank']})"
                )
            hyperparameters = artifact.get("compiler_hyperparameters", {})
            expected_hyperparameters = {
                "n_sweeps": 2,
                "prior_gamma": 0.01,
                "alpha_awt": 0.5,
                "chunk_size": 32,
                "diagonal_prior_strength": 0.0,
                "requested_bridge_rank": 128,
                "requested_sparse_budget": 524288,
                "quantize_components": True,
                "quantization_schedule": expected_quantization_schedule,
                "cg_threshold": expected_cg_threshold,
                "cg_iterations": expected_cg_iterations,
            }
            legacy_hyperparameter_defaults = {
                "quantization_schedule": "in_sweep",
                "cg_threshold": 4096,
                "cg_iterations": 6,
            }
            for key, value in expected_hyperparameters.items():
                actual_value = hyperparameters.get(key, legacy_hyperparameter_defaults.get(key))
                if actual_value != value:
                    raise ValueError(
                        f"layer{index}.{projection}: compiler setting {key}="
                        f"{actual_value!r}, expected {value!r}"
                    )
            if hyperparameters.get("realized_sparse_count") != projection_report["sparse_coefficients"]:
                raise ValueError(
                    f"layer{index}.{projection}: realized sparse count disagrees with serialized values"
                )
            if artifact.get("quantization_bits") != 4 or artifact.get("core_quantization_bits") != 4:
                raise ValueError(f"layer{index}.{projection}: expected 4-bit component quantization")
            precision = artifact.get("precision_diagnostics")
            expected_precision_components = {
                "sparse_shield", "bridge_A", "bridge_B",
                "mpo_site_0", "mpo_site_1", "mpo_site_2", "mpo_site_3",
            }
            if not isinstance(precision, dict) or set(precision) != expected_precision_components:
                raise ValueError(
                    f"layer{index}.{projection}: missing per-component quantization diagnostics"
                )
            component_numels = {
                "sparse_shield": shape[0] * shape[1],
                "bridge_A": shape[0] * expected["r_bridge"],
                "bridge_B": expected["r_bridge"] * shape[1],
                **{
                    f"mpo_site_{site}": int(math.prod(core_shape))
                    for site, core_shape in enumerate(projection_report["core_shapes"])
                },
            }
            for component, diagnostic in precision.items():
                if not isinstance(diagnostic, dict):
                    raise ValueError(f"layer{index}.{projection}.{component}: malformed precision audit")
                if diagnostic.get("group_size") != 64 or diagnostic.get("input_coefficients") != component_numels[component]:
                    raise ValueError(f"layer{index}.{projection}.{component}: quantizer metadata mismatch")
                for key in (
                    "q4_clipped_code_count", "q4_nonzero_input_rounded_to_zero_count",
                    "q4_positive_max_code_count", "q4_negative_min_code_count",
                    "fp16_nonzero_value_underflow_to_zero_count", "fp16_subnormal_count",
                    "fp16_overflow_count", "fp16_nonfinite_count",
                ):
                    value = diagnostic.get(key)
                    if not isinstance(value, int) or value < 0 or value > component_numels[component]:
                        raise ValueError(f"layer{index}.{projection}.{component}: invalid {key}")
                if diagnostic["q4_clipped_code_count"] != 0:
                    raise ValueError(f"layer{index}.{projection}.{component}: Q4 clamp clipped values")
                if diagnostic["fp16_overflow_count"] or diagnostic["fp16_nonfinite_count"]:
                    raise ValueError(f"layer{index}.{projection}.{component}: FP16 overflow/non-finite values")
                for key in ("input_max_abs", "dequantized_max_abs", "stored_fp16_max_abs"):
                    value = diagnostic.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"layer{index}.{projection}.{component}: invalid {key}")
            projection_report["precision_diagnostics"] = precision
            projection_reports[projection] = projection_report
            projection_coefficients += projection_reports[projection][
                "independent_numerical_coefficients"
            ]
            if reconstruct:
                weight = reconstruct_from_artifact(
                    artifacts[projection], device=reconstruction_device,
                    dtype=torch.float32,
                )
                if tuple(weight.shape) != shape or not torch.isfinite(weight).all():
                    raise ValueError(f"Non-finite reconstruction for layer {index} {projection}")
                del weight
        for norm_name in ("input_layernorm", "post_attention_layernorm"):
            norm = checkpoint.get(norm_name)
            if not torch.is_tensor(norm) or tuple(norm.shape) != (4096,):
                raise ValueError(f"Missing or malformed {norm_name} in layer {index}")
            _finite_tensor(f"layer{index}.{norm_name}", norm)
            layer_norm_coefficients += norm.numel()
        size = path.stat().st_size
        manifest_size = manifest.get("checkpoint_file_sizes", {}).get(path.name)
        if int(manifest_size) != size:
            raise ValueError(f"Serialized size mismatch for {path.name}")
        manifest_sha = manifest.get("checkpoint_file_sha256", {}).get(path.name)
        if not manifest_sha:
            raise ValueError(f"Missing serialized hash for {path.name}")
        if manifest_sha != file_sha256:
            raise ValueError(f"Serialized hash mismatch for {path.name}")
        serialized_bytes += size
        layers.append({
            "layer": index,
            "file": path.name,
            "bytes": size,
            "sha256": file_sha256,
            "projections": projection_reports,
        })

    dense_projection_coefficients_per_layer = sum(
        int(shape[0] * shape[1]) for shape in PROJECTION_SHAPES.values()
    )
    uncompressed_projection_coefficients = (
        32 - len(compressed_layers)
    ) * dense_projection_coefficients_per_layer
    # Layer norms are dense in both the compressed and uncompressed prefix
    # evaluation; count the full model's 64 RMSNorm vectors exactly once.
    layer_norm_coefficients = 32 * 2 * 4096
    independent = (
        projection_coefficients + uncompressed_projection_coefficients
        + layer_norm_coefficients + UNCHANGED_COEFFICIENTS
    )
    reduction = 1.0 - independent / DENSE_LLAMA2_7B_COEFFICIENTS
    ratio = DENSE_LLAMA2_7B_COEFFICIENTS / independent
    if not all(math.isfinite(value) for value in (reduction, ratio)):
        raise RuntimeError("Non-finite coefficient accounting result")
    return {
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "run_fingerprint": manifest["run_fingerprint"],
        "layers": layers,
        "compressed_layer_count": len(compressed_layers),
        "compressed_layers": compressed_layers,
        "projection_coefficients": int(projection_coefficients),
        "layer_norm_coefficients": int(layer_norm_coefficients),
        "uncompressed_dense_projection_coefficients": int(uncompressed_projection_coefficients),
        "unchanged_embedding_head_final_norm_coefficients": UNCHANGED_COEFFICIENTS,
        "independent_numerical_coefficients": int(independent),
        "dense_reference_coefficients": DENSE_LLAMA2_7B_COEFFICIENTS,
        "structural_parameter_reduction": reduction,
        "parameter_compression_ratio": ratio,
        "serialized_layer_checkpoint_bytes": serialized_bytes,
        "serialized_intermediate_artifact_bytes": intermediate_bytes,
        "reconstruction_checked": bool(reconstruct),
    }
