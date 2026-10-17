"""Matched-coefficient functional module comparisons for MLP and attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nox.function_aware_attention import (
    _streamed_covariance_cholesky,
    apply_factorized_attention,
    attention_dense_coefficient_count,
    build_factorized_attention,
)
from nox.function_aware_tn import (
    activation_weighted_svd_factors,
    activation_weighted_svd_factors_from_cholesky,
    apply_factorized_swiglu,
    apply_selected_swiglu,
    build_factorized_swiglu,
    build_selected_swiglu,
    budgeted_swiglu_rank,
    ordinary_svd_factors,
    rank_allocations,
    regularized_covariance_cholesky,
    swiglu_dense_coefficient_count,
    swiglu_factor_coefficient_count,
)
from nox.grouped_swiglu_tn import (
    apply_grouped_swiglu,
    build_grouped_swiglu_artifact,
)


def output_metrics(reference: torch.Tensor, candidate: torch.Tensor):
    ref = reference.float()
    cand = candidate.float()
    delta = cand - ref
    ref_norm = torch.linalg.vector_norm(ref).clamp_min(1e-30)
    cand_norm = torch.linalg.vector_norm(cand)
    return {
        "relative_error": float(torch.linalg.vector_norm(delta) / ref_norm),
        "cosine_similarity": float(F.cosine_similarity(ref.T, cand.T, dim=1).mean()),
        "norm_ratio": float(cand_norm / ref_norm),
    }


def _dense_mlp(weights, inputs):
    output = torch.empty(
        weights["down_proj"].shape[0], inputs.shape[1],
        device=inputs.device, dtype=torch.float32,
    )
    for start in range(0, inputs.shape[1], 512):
        stop = min(inputs.shape[1], start + 512)
        x = inputs[:, start:stop]
        output[:, start:stop] = weights["down_proj"] @ (
            F.silu(weights["gate_proj"] @ x) * (weights["up_proj"] @ x)
        )
    return output


def _low_rank_artifact(
    weights, ranks, mode, fit_inputs, seed=0, *,
    input_cholesky=None, hidden_cholesky=None,
):
    factors = {}
    for index, (name, rank) in enumerate(zip(("gate_proj", "up_proj", "down_proj"), ranks)):
        if mode == "ordinary_svd":
            factors[name] = ordinary_svd_factors(weights[name], rank, seed=seed + index)
        elif mode == "activation_weighted_svd":
            if name == "down_proj":
                calibration_inputs = None if hidden_cholesky is not None else (
                    F.silu(weights["gate_proj"] @ fit_inputs)
                    * (weights["up_proj"] @ fit_inputs)
                )
                chol = hidden_cholesky
            else:
                calibration_inputs = fit_inputs
                chol = input_cholesky
            factors[name] = (
                activation_weighted_svd_factors_from_cholesky(
                    weights[name], chol, rank, seed=seed + index,
                )
                if chol is not None else activation_weighted_svd_factors(
                    weights[name], calibration_inputs, rank, ridge_ratio=0.01,
                    seed=seed + index,
                )
            )
        else:
            raise ValueError(mode)
    return {
        "codec_type": "function_aware_factorized_swiglu_v1",
        "construction": mode,
        "ranks": tuple(ranks),
        "gate_left": factors["gate_proj"][0],
        "gate_right": factors["gate_proj"][1],
        "up_left": factors["up_proj"][0],
        "up_right": factors["up_proj"][1],
        "down_left": factors["down_proj"][0],
        "down_right": factors["down_proj"][1],
        "coefficient_count": swiglu_factor_coefficient_count(
            weights["gate_proj"].shape[1], weights["gate_proj"].shape[0], ranks,
        ),
        "dense_residual": False,
    }


def _evaluate_swiglu_candidate(weights, artifact, splits):
    result = {}
    for name, value in splits.items():
        inputs = value["mlp_input"]
        residual = value.get("residual")
        target = value.get("mlp_target")
        if target is None:
            target = _dense_mlp(weights, inputs)
        prediction = apply_factorized_swiglu(artifact, inputs)
        record = {"mlp_output": output_metrics(target, prediction)}
        if residual is not None:
            record["residual_block_output"] = output_metrics(
                residual + target, residual + prediction,
            )
        result[name] = record
    return result


@torch.no_grad()
def benchmark_swiglu_module(
    weights,
    fit_inputs,
    selection_inputs,
    heldout_inputs,
    *,
    fit_residual=None,
    selection_residual=None,
    heldout_residual=None,
    budgets=(0.75, 0.50, 0.25),
    include_grouped=True,
    seed=0,
):
    """Compare classical and functional MLP factors at exact scalar budgets."""
    gate, up, down = (weights[name] for name in ("gate_proj", "up_proj", "down_proj"))
    d_model = gate.shape[1]
    intermediate = gate.shape[0]
    if up.shape != gate.shape or down.shape != (d_model, intermediate):
        raise ValueError("incompatible SwiGLU matrices")
    fit_x = fit_inputs.detach().float()
    input_covariance = torch.zeros(
        d_model, d_model, device=fit_x.device, dtype=torch.float32,
    )
    for start in range(0, fit_x.shape[1], 512):
        x_chunk = fit_x[:, start:start + 512]
        input_covariance.addmm_(
            x_chunk, x_chunk.T, beta=1.0,
            alpha=1.0 / fit_x.shape[1],
        )
    input_cholesky, input_ridge = regularized_covariance_cholesky(input_covariance, 0.01)
    hidden_covariance = torch.zeros(
        intermediate, intermediate, device=fit_x.device, dtype=torch.float32,
    )
    for start in range(0, fit_x.shape[1], 512):
        x_chunk = fit_x[:, start:start + 512]
        hidden_chunk = F.silu(gate @ x_chunk) * (up @ x_chunk)
        hidden_covariance.addmm_(
            hidden_chunk, hidden_chunk.T, beta=1.0,
            alpha=1.0 / fit_x.shape[1],
        )
    hidden_cholesky, hidden_ridge = regularized_covariance_cholesky(
        hidden_covariance, 0.01,
    )
    selection_x = selection_inputs.detach().float()
    heldout_x = heldout_inputs.detach().float()
    splits = {
        "fit": {"mlp_input": fit_x, "residual": fit_residual},
        "internal_selection": {"mlp_input": selection_x, "residual": selection_residual},
        "heldout": {"mlp_input": heldout_x, "residual": heldout_residual},
    }
    for split in splits.values():
        split["mlp_target"] = _dense_mlp(weights, split["mlp_input"])
    results = {
        "module": "swiglu_mlp",
        "dense_coefficients": swiglu_dense_coefficient_count(d_model, intermediate),
        "fit_samples": int(fit_inputs.shape[1]),
        "selection_samples": int(selection_inputs.shape[1]),
        "heldout_samples": int(heldout_inputs.shape[1]),
        "post_compression_optimization": False,
        "uses_external_teacher_at_inference": False,
        "streaming_covariance_statistics": {
            "input_covariance_shape": list(input_covariance.shape),
            "hidden_covariance_shape": list(hidden_covariance.shape),
            "hidden_activation_chunk_size": 512,
            "input_covariance_chunk_size": 512,
            "input_ridge_value": float(input_ridge),
            "hidden_ridge_value": float(hidden_ridge),
            "readout_sketch_streams_all_fit_tokens": True,
        },
        "budgets": {},
    }
    for fraction in budgets:
        budget = budgeted_swiglu_rank(d_model, intermediate, fraction)
        ranks = budget["ranks"]
        trial_candidates = {
            "ordinary_svd": [],
            "activation_weighted_svd": [],
            "function_aware": [],
            "modegpt_inspired_shared_neuron_selection": [],
        }
        allocation_trials = rank_allocations(sum(ranks))
        allocation_trials = {
            name: allocation_trials[name]
            for name in ("balanced", "more_down")
            if name in allocation_trials
        }
        for allocation, trial_ranks in allocation_trials.items():
            trial_candidates["ordinary_svd"].append((
                {"allocation": allocation},
                _low_rank_artifact(weights, trial_ranks, "ordinary_svd", fit_x, seed),
            ))
            trial_candidates["activation_weighted_svd"].append((
                {"allocation": allocation, "ridge_ratio": 0.01},
                _low_rank_artifact(
                    weights, trial_ranks, "activation_weighted_svd", fit_x, seed,
                    input_cholesky=input_cholesky, hidden_cholesky=hidden_cholesky,
                ),
            ))
            trial_candidates["function_aware"].append((
                {"allocation": allocation, "ridge_ratio": 0.01},
                build_factorized_swiglu(
                    gate, up, down, fit_x, trial_ranks, construction="function_aware",
                    ridge_ratio=0.01, storage_dtype=torch.float32, seed=seed,
                    sketch_size=min(fit_x.shape[1], trial_ranks[2] + 64),
                    sketch_chunk_size=512,
                    fit_input_cholesky=input_cholesky,
                ),
            ))
        for ridge_ratio in (0.001, 0.01, 0.1):
            trial_candidates["modegpt_inspired_shared_neuron_selection"].append((
                {"ridge_ratio": ridge_ratio},
                build_selected_swiglu(
                    gate, up, down, fit_x, budget["coefficient_budget"],
                    ridge_ratio=ridge_ratio, storage_dtype=torch.float32,
                ),
            ))

        artifacts = {}
        selected_trial_metadata = {}
        for method, trials in trial_candidates.items():
            ranked = []
            for configuration, artifact in trials:
                if method == "modegpt_inspired_shared_neuron_selection":
                    predicted = apply_selected_swiglu(artifact, selection_x)
                else:
                    predicted = apply_factorized_swiglu(artifact, selection_x)
                selection_error = output_metrics(
                    splits["internal_selection"]["mlp_target"], predicted,
                )["relative_error"]
                ranked.append((selection_error, configuration, artifact))
            ranked.sort(key=lambda item: item[0])
            selected_error, selected_configuration, selected_artifact = ranked[0]
            artifacts[method] = selected_artifact
            selected_trial_metadata[method] = {
                "selected_configuration": selected_configuration,
                "selected_internal_selection_relative_error": selected_error,
                "selection_trials": [
                    {"configuration": configuration, "relative_error": error}
                    for error, configuration, _ in ranked
                ],
            }
        if include_grouped:
            grouped_rank = min(
                intermediate, d_model,
                budget["coefficient_budget"] // (3 * (intermediate + d_model)),
            )
            if grouped_rank <= 0:
                raise ValueError("grouped TN cannot fit the requested coefficient budget")
            grouped, _, _ = build_grouped_swiglu_artifact(
                gate, up, down, fit_x, groups=1, rank=grouped_rank,
                storage_dtype=torch.float32,
            )
            artifacts["previous_grouped_tn"] = grouped
        budget_record = {**budget, "candidates": {}}
        for name, artifact in artifacts.items():
            if name == "previous_grouped_tn":
                metrics = {}
                for split_name, split in splits.items():
                    x = split["mlp_input"]
                    target = split["mlp_target"]
                    prediction = apply_grouped_swiglu(artifact, x)
                    record = {"mlp_output": output_metrics(target, prediction)}
                    if split.get("residual") is not None:
                        record["residual_block_output"] = output_metrics(
                            split["residual"] + target, split["residual"] + prediction,
                        )
                    metrics[split_name] = record
            elif name == "modegpt_inspired_shared_neuron_selection":
                metrics = {}
                for split_name, split in splits.items():
                    x = split["mlp_input"]
                    target = split["mlp_target"]
                    prediction = apply_selected_swiglu(artifact, x)
                    record = {"mlp_output": output_metrics(target, prediction)}
                    if split.get("residual") is not None:
                        record["residual_block_output"] = output_metrics(
                            split["residual"] + target, split["residual"] + prediction,
                        )
                    metrics[split_name] = record
            else:
                metrics = _evaluate_swiglu_candidate(weights, artifact, splits)
            count = int(artifact["coefficient_count"])
            budget_record["candidates"][name] = {
                "ranks": list(artifact["ranks"]) if "ranks" in artifact else None,
                "selected_neuron_count": artifact.get("neuron_count"),
                "coefficient_count": count,
                "actual_fraction": count / results["dense_coefficients"],
                "parameter_compression_ratio": results["dense_coefficients"] / count,
                "fp32_payload_bytes": count * 4,
                "fp16_payload_bytes": count * 2,
                "uses_gradients": False,
                "dense_residual": False,
                "internal_selection": selected_trial_metadata.get(name),
                "construction_metadata": {
                    key: artifact[key]
                    for key in (
                        "construction", "down_fit", "selection_method", "down_fit_solver",
                        "fit_token_sketch_size", "fit_sketch_method", "ridge_ratio",
                        "ridge_value", "construction_objective", "groups", "rank",
                    )
                    if key in artifact
                },
                "metrics": metrics,
            }
        results["budgets"][str(float(fraction))] = budget_record
    return results


def _apply_attention(weights, artifact, inputs, *, n_heads, n_kv_heads, seq_len):
    return apply_factorized_attention(
        artifact, inputs, n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len,
    )


@torch.no_grad()
def benchmark_attention_module(
    weights,
    fit_inputs,
    fit_targets,
    selection_inputs,
    selection_targets,
    heldout_inputs,
    heldout_targets,
    *,
    n_heads,
    n_kv_heads,
    seq_len,
    fit_residual=None,
    selection_residual=None,
    heldout_residual=None,
    downstream_fn=None,
    budgets=(0.75, 0.50, 0.25),
    seed=0,
):
    """Compare complete-attention output candidates at matched coefficients."""
    dense_count = attention_dense_coefficient_count(weights)
    fit_input_cholesky, input_ridge = _streamed_covariance_cholesky(fit_inputs)
    result = {
        "module": "complete_attention_output",
        "dense_coefficients": dense_count,
        "fit_samples": int(fit_inputs.shape[1]),
        "selection_samples": int(selection_inputs.shape[1]),
        "heldout_samples": int(heldout_inputs.shape[1]),
        "post_compression_optimization": False,
        "streaming_covariance_statistics": {
            "fit_input_covariance_chunk_size": 512,
            "fit_input_ridge_value": float(input_ridge),
            "fit_token_count": int(fit_inputs.shape[1]),
        },
        "budgets": {},
    }
    splits = {
        "fit": (fit_inputs, fit_targets, fit_residual),
        "internal_selection": (selection_inputs, selection_targets, selection_residual),
        "heldout": (heldout_inputs, heldout_targets, heldout_residual),
    }
    for fraction in budgets:
        parameter_budget = int(dense_count * float(fraction))
        per_rank_cost = sum(
            weights[name].shape[0] + weights[name].shape[1]
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        rank_total = parameter_budget // per_rank_cost
        base, remainder = divmod(rank_total, 4)
        ranks = tuple(base + (1 if i < remainder else 0) for i in range(4))
        if min(ranks) < 1:
            raise ValueError("attention coefficient budget is too small for rank one")
        artifacts = {
            name: build_factorized_attention(
                weights, fit_inputs, fit_targets, ranks,
                n_heads=n_heads, n_kv_heads=n_kv_heads, seq_len=seq_len,
                construction=name, ridge_ratio=0.01, seed=seed,
                storage_dtype=torch.float32,
                fit_input_cholesky=fit_input_cholesky,
            )
            for name in (
                "ordinary_svd", "activation_weighted_svd", "function_aware",
                "module_output_readout",
            )
        }
        budget_record = {
            "coefficient_budget": parameter_budget,
            "ranks": list(ranks),
            "candidates": {},
        }
        for name, artifact in artifacts.items():
            metric_records = {}
            for split_name, (inputs, targets, residual) in splits.items():
                prediction = _apply_attention(
                    weights, artifact, inputs, n_heads=n_heads,
                    n_kv_heads=n_kv_heads, seq_len=seq_len,
                )
                metrics = {"attention_output": output_metrics(targets, prediction)}
                if residual is not None:
                    metrics["residual_block_output"] = output_metrics(
                        residual + targets, residual + prediction,
                    )
                    if downstream_fn is not None:
                        metrics["full_residual_block_output"] = output_metrics(
                            downstream_fn(residual + targets),
                            downstream_fn(residual + prediction),
                        )
                metric_records[split_name] = metrics
            count = int(artifact["coefficient_count"])
            budget_record["candidates"][name] = {
                "ranks": list(artifact["ranks"]),
                "coefficient_count": count,
                "actual_fraction": count / dense_count,
                "parameter_compression_ratio": dense_count / count,
                "fp32_payload_bytes": count * 4,
                "fp16_payload_bytes": count * 2,
                "uses_gradients": False,
                "dense_residual": False,
                "metrics": metric_records,
                "module_output_readout": artifact["module_output_readout"],
            }
        result["budgets"][str(float(fraction))] = budget_record
    return result
