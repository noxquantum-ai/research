#!/usr/bin/env python3
"""Real-model GPU qualification used before the expensive NEC compilation."""

from __future__ import annotations

import gc
import json
import math
import numbers
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.benchmark_day4_perplexity import compute_perplexity, load_wikitext2_test_text
from benchmarks.evaluate_high_fidelity_mmlu import (
    install_pinned_mmlu_loader,
    mmlu_sample_identity,
    summarize_mmlu_harness,
)
from benchmarks.evaluate_high_fidelity_nec import load_model
from nox.compiler import nox_hybrid_dmrg_decompose, reconstruct_from_artifact
from nox.evaluation import validate_mpo_artifact
from nox.utils import get_modes_for_projection
from nox.mmlu import load_full_mmlu
from nox.mmlu import prepare_example


class BatchEquivalenceError(RuntimeError):
    """The selected evaluator batch differs numerically from serial scoring."""


def _cuda_identity() -> list[dict]:
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            "free_memory_bytes_at_qualification": torch.cuda.mem_get_info(index)[0],
            "compute_capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ]


def _release_cuda() -> None:
    gc.collect()
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            torch.cuda.empty_cache()


def _candidate_batches(initial: int) -> list[int]:
    values = []
    current = initial
    while current >= 1:
        if current not in values:
            values.append(current)
        if current == 1:
            break
        current = max(1, current // 2)
    return values


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _metric_summary(details: dict, ppl: float, scored: int) -> dict:
    return {
        "perplexity": ppl,
        "mean_nll": details["mean_nll"],
        "next_token_accuracy": details["next_token_accuracy"],
        "input_tokens": details["input_tokens"],
        "scored_tokens": scored,
        "token_ids_sha256": details["token_ids_sha256"],
        "window_specification": details["window_specification"],
        "forward_passes": details["forward_passes"],
    }


def _numeric_leaves(value) -> list[float]:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numeric_leaves(item)]
    return []


def _assert_response_tree_close(left, right, tolerance: float, path: str = "response") -> float:
    """Compare a harness response tree, tolerating only small numeric drift."""
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            raise BatchEquivalenceError(f"Boolean response changed at {path}")
        return 0.0
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        left_value, right_value = float(left), float(right)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise BatchEquivalenceError(f"Non-finite response value at {path}")
        delta = abs(left_value - right_value)
        if delta > tolerance:
            raise BatchEquivalenceError(
                f"Response value at {path} differs by {delta:.6g}, exceeding {tolerance}"
            )
        return delta
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise BatchEquivalenceError(f"Response shape changed at {path}")
        return max(
            (_assert_response_tree_close(a, b, tolerance, f"{path}[{index}]")
             for index, (a, b) in enumerate(zip(left, right))),
            default=0.0,
        )
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict) or left.keys() != right.keys():
            raise BatchEquivalenceError(f"Response mapping changed at {path}")
        return max(
            (_assert_response_tree_close(left[key], right[key], tolerance, f"{path}.{key}")
             for key in left),
            default=0.0,
        )
    if left != right:
        raise BatchEquivalenceError(f"Non-numeric response changed at {path}")
    return 0.0


def compare_mmlu_smokes(serial: dict, batched: dict, subjects: list[str], tolerance: float = 1e-4) -> dict:
    """Require identical prompts and predictions, allowing small score roundoff."""
    if mmlu_sample_identity(serial, subjects) != mmlu_sample_identity(batched, subjects):
        raise BatchEquivalenceError("Serial and batched MMLU smoke prompts differ")
    checked = 0
    maximum_absolute_score_delta = 0.0
    for subject in subjects:
        task = f"mmlu_{subject}"
        left_rows = serial.get("samples", {}).get(task, [])
        right_rows = batched.get("samples", {}).get(task, [])
        if len(left_rows) != 1 or len(right_rows) != 1:
            raise BatchEquivalenceError(f"MMLU smoke did not return one row for {task}")
        left, right = left_rows[0], right_rows[0]
        filtered_delta = _assert_response_tree_close(
            left.get("filtered_resps"), right.get("filtered_resps"), tolerance,
            f"{task}.filtered_resps",
        )
        response_delta = _assert_response_tree_close(
            left.get("resps"), right.get("resps"), tolerance, f"{task}.resps",
        )
        left_scores = _numeric_leaves(left.get("resps"))
        right_scores = _numeric_leaves(right.get("resps"))
        if len(left_scores) != 4 or len(left_scores) != len(right_scores):
            raise BatchEquivalenceError(f"MMLU score vectors differ in shape for {task}")
        if max(range(4), key=left_scores.__getitem__) != max(range(4), key=right_scores.__getitem__):
            raise BatchEquivalenceError(f"MMLU predicted answer choice changed under batching for {task}")
        if left.get("metrics") != right.get("metrics"):
            raise BatchEquivalenceError(f"MMLU task metrics changed under batching for {task}")
        metric_names = left.get("metrics", [])
        for metric in metric_names:
            if left.get(metric) != right.get(metric):
                raise BatchEquivalenceError(f"MMLU metric {metric} changed under batching for {task}")
        delta = max(
            filtered_delta,
            response_delta,
            *(abs(a - b) for a, b in zip(left_scores, right_scores)),
        )
        maximum_absolute_score_delta = max(maximum_absolute_score_delta, delta)
        checked += len(left_scores)
    return {
        "status": "PASS",
        "questions": len(subjects),
        "numerical_likelihood_values_compared": checked,
        "maximum_absolute_likelihood_delta": maximum_absolute_score_delta,
        "maximum_absolute_likelihood_delta_tolerance": tolerance,
        "filtered_responses_within_tolerance": True,
        "predicted_answer_choices_exactly_equal": True,
        "task_metrics_exactly_equal": True,
        "prompt_identities_exactly_equal": True,
    }


def qualify_gpu(
    *, run_root: Path, source_commit: str, run_config_sha256: str,
    requested_ppl_batch: int, requested_mmlu_batch: int,
    compression: dict, calibration_tokens: int, model_id: str, model_revision: str,
) -> dict:
    """Load the pinned dense model and prove batching/harness behavior on this GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU qualification requires CUDA")
    target = run_root / "gpu_qualification.json"
    hardware = _cuda_identity()
    batch_request = {
        "ppl": requested_ppl_batch,
        "mmlu": requested_mmlu_batch,
    }
    memory_caps = {
        "gpu_per_device_gib": int(os.environ["NEC_EVAL_GPU_MEMORY_GIB"]),
        "cpu_gib": int(os.environ["NEC_EVAL_CPU_MEMORY_GIB"]),
    }
    if target.is_file():
        existing = json.loads(target.read_text())
        if (
            existing.get("status") == "PASS"
            and existing.get("source_commit") == source_commit
            and existing.get("run_config_sha256") == run_config_sha256
            and existing.get("hardware") == hardware
            and existing.get("requested_batch_sizes") == batch_request
            and existing.get("evaluation_memory_caps_gib") == memory_caps
        ):
            return existing

    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    model, tokenizer, injected, compatibility, diagnostics = load_model(
        None, offload_folder=run_root / "preflight_offload", variant="dense",
    )
    if injected or diagnostics:
        raise RuntimeError("Dense qualification unexpectedly injected compressed weights")
    text, dataset = load_wikitext2_test_text(return_metadata=True)
    device = str(model.get_input_embeddings().weight.device)

    serial_ppl, serial_scored, serial = compute_perplexity(
        model, tokenizer, text, seq_len=2048, stride=512, max_eval_tokens=10_000,
        device=device, return_details=True, verify_reference_loss=True, batch_size=1,
    )
    if serial_scored != 9_999:
        raise RuntimeError(f"Serial PPL qualification scored {serial_scored}, expected 9,999")

    ppl_batch = None
    batched_ppl = None
    batched_scored = None
    batched = None
    ppl_attempts = []
    for candidate in _candidate_batches(requested_ppl_batch):
        try:
            value, scored, details = compute_perplexity(
                model, tokenizer, text, seq_len=2048, stride=512, max_eval_tokens=10_000,
                device=device, return_details=True, verify_reference_loss=True,
                batch_size=candidate,
            )
            if scored != serial_scored:
                raise RuntimeError("Batched PPL qualification changed the scored-token count")
            if details["token_ids_sha256"] != serial["token_ids_sha256"]:
                raise RuntimeError("Batched PPL qualification changed the token stream")
            top1_agreement = float((
                details["top1_predictions"] == serial["top1_predictions"]
            ).float().mean().item())
            if top1_agreement < 0.999:
                raise BatchEquivalenceError(
                    f"Batched PPL qualification dense top-1 agreement {top1_agreement:.6f} < 0.999"
                )
            if abs(details["mean_nll"] - serial["mean_nll"]) > 1e-4:
                raise BatchEquivalenceError("Batched PPL qualification changed mean NLL beyond 1e-4")
            if abs(value - serial_ppl) > 1e-4 * max(1.0, serial_ppl):
                raise BatchEquivalenceError(
                    "Batched PPL qualification changed perplexity beyond 1e-4 relative"
                )
            ppl_batch, batched_ppl, batched_scored, batched = candidate, value, scored, details
            ppl_attempts.append({"batch_size": candidate, "status": "PASS"})
            break
        except BaseException as error:
            ppl_attempts.append({
                "batch_size": candidate,
                "status": (
                    "OOM" if _is_oom(error)
                    else "NUMERICAL_MISMATCH" if isinstance(error, BatchEquivalenceError)
                    else "FAIL"
                ),
                "error": f"{type(error).__name__}: {error}",
            })
            _release_cuda()
            if not _is_oom(error) and not isinstance(error, BatchEquivalenceError):
                raise
    if ppl_batch is None or batched is None or batched_ppl is None or batched_scored is None:
        raise RuntimeError("No PPL batch size passed GPU qualification")

    # Scan every pinned test prompt with its five same-subject demonstrations.
    # Then exercise the selected batch on the longest actual prompt and on the
    # full 4096-token model context ceiling.
    splits, pinned_data = load_full_mmlu()
    subjects = sorted({row["subject"] for row in splits["test"]})
    if len(subjects) != 57:
        raise RuntimeError("Pinned MMLU qualification did not find 57 subjects")
    demonstrations = {subject: [] for subject in subjects}
    for row in splits["dev"]:
        demonstrations[row["subject"]].append(row)
    longest_example = None
    longest_length = -1
    minimum_shots = 5
    for row in splits["test"]:
        example = prepare_example(
            tokenizer, row, demonstrations[row["subject"]], max_context=4096,
        )
        if len(example["input_ids"]) > longest_length:
            longest_example = example
            longest_length = len(example["input_ids"])
        minimum_shots = min(minimum_shots, int(example["shots"]))
    if longest_example is None or longest_length >= 4096:
        raise RuntimeError("Could not prepare a valid MMLU maximum-context prompt")
    if minimum_shots != 5:
        raise RuntimeError(
            "At least one pinned MMLU prompt cannot retain all five demonstrations within 4096 tokens"
        )
    model_input_device = model.get_input_embeddings().weight.device
    longest_prompt = torch.tensor(
        longest_example["input_ids"], dtype=torch.long, device=model_input_device,
    )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    if encoded.numel() < 4096:
        encoded = encoded.repeat(math.ceil(4096 / encoded.numel()))
    max_context = encoded[:4096].to(model_input_device)
    mmlu_batch = None
    mmlu_attempts = []
    for candidate in _candidate_batches(requested_mmlu_batch):
        try:
            with torch.inference_mode():
                for sequence in (longest_prompt, max_context):
                    output = model(
                        input_ids=sequence.unsqueeze(0).expand(candidate, -1),
                        attention_mask=torch.ones(
                            (candidate, sequence.numel()), dtype=torch.long, device=sequence.device,
                        ),
                        use_cache=False,
                    ).logits[:, -1, :]
                    if output.shape != (candidate, 32_000) or not torch.isfinite(output).all():
                        raise RuntimeError("MMLU qualification produced invalid logits")
                    del output
            mmlu_batch = candidate
            mmlu_attempts.append({"batch_size": candidate, "status": "PASS"})
            break
        except BaseException as error:
            mmlu_attempts.append({
                "batch_size": candidate,
                "status": "OOM" if _is_oom(error) else "FAIL",
                "error": f"{type(error).__name__}: {error}",
            })
            _release_cuda()
            if not _is_oom(error):
                raise
    if mmlu_batch is None:
        raise RuntimeError("No maximum-context MMLU batch size passed GPU qualification")

    import datasets
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    calls, original_loader = install_pinned_mmlu_loader(datasets)
    mmlu_equivalence_attempts = []
    selected_mmlu_equivalence = None
    serial_smoke = None
    selected_smoke = None
    mmlu_batch_equivalence = None
    batched_lm = None
    try:
        serial_lm = HFLM(
            pretrained=model, tokenizer=tokenizer, batch_size=1,
            max_length=4096, logits_cache=True,
        )
        serial_smoke = evaluator.simple_evaluate(
            model=serial_lm, tasks=["mmlu"], num_fewshot=5, batch_size=1,
            device="cuda:0", limit=1, bootstrap_iters=0, check_integrity=True,
            log_samples=True, apply_chat_template=False, fewshot_as_multiturn=False,
            random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
            fewshot_random_seed=1234,
        )
        if serial_smoke is None:
            raise RuntimeError("Serial MMLU harness qualification returned no result")
        for candidate in _candidate_batches(mmlu_batch):
            if candidate == 1:
                selected_smoke = serial_smoke
                selected_mmlu_equivalence = compare_mmlu_smokes(
                    serial_smoke, serial_smoke, subjects,
                )
                mmlu_equivalence_attempts.append({"batch_size": 1, "status": "PASS"})
                mmlu_batch = 1
                break
            try:
                batched_lm = HFLM(
                    pretrained=model, tokenizer=tokenizer, batch_size=candidate,
                    max_length=4096, logits_cache=True,
                )
                candidate_smoke = evaluator.simple_evaluate(
                    model=batched_lm, tasks=["mmlu"], num_fewshot=5,
                    batch_size=candidate, device="cuda:0", limit=1,
                    bootstrap_iters=0, check_integrity=True, log_samples=True,
                    apply_chat_template=False, fewshot_as_multiturn=False,
                    random_seed=0, numpy_random_seed=1234, torch_random_seed=1234,
                    fewshot_random_seed=1234,
                )
                if candidate_smoke is None:
                    raise RuntimeError("Batched MMLU harness qualification returned no result")
                equivalence = compare_mmlu_smokes(serial_smoke, candidate_smoke, subjects)
                selected_smoke = candidate_smoke
                selected_mmlu_equivalence = equivalence
                mmlu_equivalence_attempts.append({
                    "batch_size": candidate,
                    "status": "PASS",
                    "equivalence": equivalence,
                })
                mmlu_batch = candidate
                break
            except BaseException as error:
                mmlu_equivalence_attempts.append({
                    "batch_size": candidate,
                    "status": (
                        "OOM" if _is_oom(error)
                        else "NUMERICAL_MISMATCH" if isinstance(error, BatchEquivalenceError)
                        else "FAIL"
                    ),
                    "error": f"{type(error).__name__}: {error}",
                })
                _release_cuda()
                if not _is_oom(error) and not isinstance(error, BatchEquivalenceError):
                    raise
    finally:
        datasets.load_dataset = original_loader
    if selected_smoke is None or selected_mmlu_equivalence is None:
        raise RuntimeError("MMLU harness qualification returned no result")
    smoke_summary = summarize_mmlu_harness(selected_smoke, subjects, full_test=False)
    if smoke_summary["total_questions"] != 57:
        raise RuntimeError("MMLU harness qualification did not evaluate one item per subject")
    if not calls or any(call["revision"] != pinned_data["revision"] for call in calls):
        raise RuntimeError("MMLU harness qualification did not enforce the pinned revision")

    # Release both harness adapters before allocating the large compiler smoke
    # tensors.  They wrap the same model and otherwise keep references alive.
    del serial_lm, batched_lm, model
    _release_cuda()
    from scripts.compile_high_fidelity import PinnedWeights

    pinned_weights = PinnedWeights(model_id, model_revision)
    down_weight = pinned_weights.tensor("model.layers.0.mlp.down_proj.weight").to("cuda:0")
    del pinned_weights
    input_dimension = int(down_weight.shape[1])
    output_dimension = int(down_weight.shape[0])
    modes = get_modes_for_projection(output_dimension, input_dimension)
    random = torch.Generator(device="cuda:0").manual_seed(20261019)
    smoke_inputs = torch.randn(
        input_dimension, calibration_tokens, generator=random, device="cuda:0", dtype=torch.float32,
    )
    smoke_targets = down_weight @ smoke_inputs
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    compiler_started = time.monotonic()
    compiler_weight_hat, _, _, realized_ranks, artifact = nox_hybrid_dmrg_decompose(
        W=down_weight,
        X_calib=smoke_inputs,
        Y_dense=smoke_targets,
        o_modes=modes[0],
        i_modes=modes[1],
        ranks=tuple(compression["ranks"]),
        r_bridge=int(compression["bridge_rank"]),
        n_outliers=int(compression["sparse_outlier_budget"]),
        n_sweeps=int(compression["mpo_sweeps"]),
        prior_gamma=float(compression["local_fit_prior_gamma"]),
        alpha_awt=float(compression["activation_scale_exponent"]),
        chunk_size=int(compression["dmrg_chunk_size"]),
        return_artifacts=True,
        verbose=False,
        quantize_components=bool(compression["quantization"]["enabled"]),
        diagonal_prior_strength=float(compression["diagonal_prior_strength"]),
        site_order=tuple(compression["site_order"]),
    )
    artifact_audit = validate_mpo_artifact(
        "preflight.layer0.down_proj",
        artifact,
        (output_dimension, input_dimension),
        expected_ranks=tuple(compression["ranks"]),
    )
    decoded = reconstruct_from_artifact(artifact, device="cuda:0", dtype=torch.float32)
    if decoded.shape != (output_dimension, input_dimension) or not torch.isfinite(decoded).all():
        raise RuntimeError("High-Fidelity compiler capacity smoke reconstructed invalid down_proj")
    compiler_relative_consistency = float(
        torch.linalg.vector_norm(decoded - compiler_weight_hat)
        .div(torch.linalg.vector_norm(compiler_weight_hat).clamp_min(1e-12))
        .item()
    )
    if compiler_relative_consistency > 2e-3:
        raise RuntimeError(
            f"Serialized down_proj reconstruction mismatch: {compiler_relative_consistency}"
        )
    compiler_smoke = {
        "status": "PASS",
        "scope": "capacity and artifact decode only; synthetic calibration activations; no quality metric",
        "pinned_projection_weight": "model.layers.0.mlp.down_proj.weight",
        "input_shape": list(smoke_inputs.shape),
        "calibration_token_count": calibration_tokens,
        "weight_shape": [output_dimension, input_dimension],
        "o_modes": list(modes[0]),
        "i_modes": list(modes[1]),
        "requested_ranks": list(compression["ranks"]),
        "realized_ranks": list(realized_ranks),
        "independent_numerical_coefficients": artifact_audit["independent_numerical_coefficients"],
        "sparse_coefficients": int(artifact["sparse_values"].numel()),
        "finite_reconstruction": True,
        "artifact_decoder_relative_consistency": compiler_relative_consistency,
        "runtime_seconds": time.monotonic() - compiler_started,
        "peak_cuda_memory_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(torch.cuda.device_count())
        },
    }
    del down_weight, smoke_inputs, smoke_targets, compiler_weight_hat, artifact, decoded
    _release_cuda()

    record = {
        "status": "PASS",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": source_commit,
        "run_config_sha256": run_config_sha256,
        "hardware": hardware,
        "requested_batch_sizes": batch_request,
        "evaluation_memory_caps_gib": memory_caps,
        "model_tokenizer_compatibility": compatibility,
        "wikitext2_dataset": dataset,
        "ppl_batch_equivalence": {
            "status": "PASS",
            "requested_batch_size": requested_ppl_batch,
            "qualified_batch_size": ppl_batch,
            "tolerance": {
                "mean_nll_absolute": 1e-4,
                "ppl_relative": 1e-4,
                "dense_top1_agreement_minimum": 0.999,
            },
            "dense_top1_prediction_agreement": top1_agreement,
            "serial": _metric_summary(serial, serial_ppl, serial_scored),
            "batched": _metric_summary(batched, batched_ppl, batched_scored),
            "attempts": ppl_attempts,
        },
        "mmlu_gpu_qualification": {
            "status": "PASS",
            "requested_batch_size": requested_mmlu_batch,
            "qualified_batch_size": mmlu_batch,
            "maximum_context_tokens": 4096,
            "maximum_context_forward_passed": True,
            "longest_actual_prompt_tokens": longest_length,
            "minimum_prepared_shots_across_test_set": minimum_shots,
            "all_14042_test_prompts_tokenized_with_five_shots": True,
            "harness_version_initialized": True,
            "adapter_smoke_subjects": 57,
            "adapter_smoke_questions": 57,
            "batch_equivalence": selected_mmlu_equivalence,
            "batch_equivalence_attempts": mmlu_equivalence_attempts,
            "dataset": pinned_data,
            "dataset_revision_enforcement_calls": calls,
            "attempts": mmlu_attempts,
        },
        "compiler_capacity_smoke": compiler_smoke,
        "cuda_peak_memory_bytes": {
            str(index): int(torch.cuda.max_memory_allocated(index))
            for index in range(torch.cuda.device_count())
        },
    }
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(target)
    _release_cuda()
    return record
