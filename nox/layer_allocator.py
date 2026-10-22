"""Interaction-aware search over whole-layer compression choices.

The proxy here is a second-order expansion of end-to-end token loss in logit
space. It uses dense logits and the logit deltas from compressing one layer at
a time; it never treats scalar layer reconstruction errors as additive.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
import random
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FisherAllocationProxy:
    """Taylor coefficients for KL and NLL over single-layer logit deltas."""

    layers: tuple[int, ...]
    baseline_mean_nll: float
    linear_nll: tuple[float, ...]
    fisher_gram: tuple[tuple[float, ...], ...]
    scored_tokens: int

    def predict(self, compressed_layers: Iterable[int]) -> dict[str, float]:
        """Estimate KL and PPL for a set of compressed layers.

        For delta logits ``d = sum_i delta_i``, the estimates are
        ``KL ~= .5 * d.T @ (diag(p) - p p.T) @ d`` and the matching
        second-order Taylor expansion of target NLL.
        """
        selected = tuple(sorted(set(int(layer) for layer in compressed_layers)))
        unknown = set(selected) - set(self.layers)
        if unknown:
            raise ValueError(f"Unknown candidate layers in allocation: {sorted(unknown)}")
        positions = [self.layers.index(layer) for layer in selected]
        linear = sum(self.linear_nll[i] for i in positions)
        quadratic = 0.5 * sum(
            self.fisher_gram[i][j]
            for i in positions for j in positions
        )
        mean_nll = self.baseline_mean_nll + linear + quadratic
        return {
            "predicted_mean_nll": mean_nll,
            "predicted_perplexity": math.exp(min(mean_nll, 80.0)),
            "predicted_kl_nats_per_token": max(0.0, quadratic),
        }


def fit_fisher_allocation_proxy(
    dense_logits: torch.Tensor,
    singleton_logits: Mapping[int, torch.Tensor],
    targets: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
    token_chunk_size: int = 32,
) -> FisherAllocationProxy:
    """Fit a token-weighted Fisher proxy from one-layer model evaluations.

    Args:
        dense_logits: ``[tokens, vocabulary]`` dense-model logits.
        singleton_logits: Same-shaped logits for each one-layer replacement.
        targets: One target token ID per logit row.
        device: Device used for the vocabulary reductions. Logits may live on
            CPU to keep model inference memory bounded.
        token_chunk_size: Bounds temporary ``tokens x vocabulary`` tensors.

    The layer interaction term is the Fisher inner product between their
    end-to-end logit deltas. Negative off-diagonal terms represent predicted
    cancellation; positive terms represent predicted amplification.
    """
    if dense_logits.ndim != 2:
        raise ValueError("dense_logits must have shape [tokens, vocabulary]")
    if token_chunk_size < 1:
        raise ValueError("token_chunk_size must be positive")
    layers = tuple(sorted(int(layer) for layer in singleton_logits))
    if not layers:
        raise ValueError("At least one singleton layer evaluation is required")
    token_count, vocab_size = dense_logits.shape
    if token_count < 1 or targets.numel() != token_count:
        raise ValueError("targets must contain exactly one ID per logit row")
    if any(tuple(singleton_logits[layer].shape) != tuple(dense_logits.shape) for layer in layers):
        raise ValueError("Every singleton logit tensor must match the dense logits shape")
    if int(targets.min()) < 0 or int(targets.max()) >= vocab_size:
        raise ValueError("A target ID lies outside the logit vocabulary")

    n = len(layers)
    linear_sums = torch.zeros(n, dtype=torch.float64)
    gram_sums = torch.zeros((n, n), dtype=torch.float64)
    baseline_nll_sum = torch.zeros((), dtype=torch.float64)
    target_ids = targets.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    compute_device = torch.device(device)

    for begin in range(0, token_count, token_chunk_size):
        end = min(begin + token_chunk_size, token_count)
        dense = dense_logits[begin:end].to(device=compute_device, dtype=torch.float32)
        labels = target_ids[begin:end].to(compute_device)
        row_ids = torch.arange(end - begin, device=compute_device)
        log_probs = F.log_softmax(dense, dim=-1)
        probs = log_probs.exp()
        baseline_nll_sum += (-log_probs[row_ids, labels]).double().sum().cpu()

        deltas = []
        means = []
        for layer in layers:
            candidate = singleton_logits[layer][begin:end].to(
                device=compute_device, dtype=torch.float32,
            )
            delta = candidate - dense
            deltas.append(delta)
            means.append((probs * delta).sum(dim=-1))
            grad = means[-1] - delta[row_ids, labels]
            linear_sums[layers.index(layer)] += grad.double().sum().cpu()

        for i in range(n):
            for j in range(i, n):
                covariance = (
                    (probs * deltas[i] * deltas[j]).sum(dim=-1)
                    - means[i] * means[j]
                )
                value = covariance.double().sum().cpu()
                gram_sums[i, j] += value
                if i != j:
                    gram_sums[j, i] += value

    denom = float(token_count)
    return FisherAllocationProxy(
        layers=layers,
        baseline_mean_nll=float(baseline_nll_sum / denom),
        linear_nll=tuple(float(value / denom) for value in linear_sums),
        fisher_gram=tuple(
            tuple(float(value / denom) for value in row)
            for row in gram_sums
        ),
        scored_tokens=token_count,
    )


@dataclass(frozen=True)
class AllocationSearchResult:
    selected_layers: tuple[int, ...]
    ranked_allocations: tuple[dict, ...]
    strategy: str
    proxy_allocations_scored: int
    is_proxy_optimal: bool


def _allocation_key(
    proxy: FisherAllocationProxy,
    selected: Sequence[int],
    max_ppl: float | None,
) -> tuple[float, ...]:
    estimate = proxy.predict(selected)
    ppl = estimate["predicted_perplexity"]
    kl = estimate["predicted_kl_nats_per_token"]
    if max_ppl is None or ppl <= max_ppl:
        return (0.0, kl, ppl, *selected)
    # If no candidate satisfies the requested quality limit, report the best
    # predicted PPL and mark the allocation infeasible in its result row.
    return (1.0, ppl, kl, *selected)


def _rank_allocations(
    proxy: FisherAllocationProxy,
    masks: Iterable[tuple[int, ...]],
    max_ppl: float | None,
) -> list[dict]:
    rows = []
    for mask in sorted(set(tuple(sorted(mask)) for mask in masks)):
        estimate = proxy.predict(mask)
        estimate.update({
            "compressed_layers": list(mask),
            "predicted_quality_feasible": (
                max_ppl is None or estimate["predicted_perplexity"] <= max_ppl
            ),
        })
        rows.append(estimate)
    rows.sort(key=lambda row: _allocation_key(proxy, row["compressed_layers"], max_ppl))
    return rows


def _beam_proxy_masks(
    proxy: FisherAllocationProxy,
    compressed_count: int,
    *,
    beam_width: int,
    random_seed: int,
) -> tuple[set[tuple[int, ...]], int]:
    """Generate fixed-size masks with conditional Fisher costs and beam diversity."""
    layers = list(proxy.layers)
    all_masks: set[tuple[int, ...]] = set()
    states_examined = 0
    # Multiple orderings reduce dependence on numeric layer order. The beam
    # branches on exact conditional marginals from the current partial mask.
    rng = random.Random(random_seed)
    orders = [layers, list(reversed(layers))]
    sensitivity_order = sorted(
        layers,
        key=lambda layer: proxy.predict((layer,))["predicted_kl_nats_per_token"],
    )
    orders.append(sensitivity_order)
    for _ in range(5):
        order = layers.copy()
        rng.shuffle(order)
        orders.append(order)

    for order in orders:
        beam: list[tuple[int, ...]] = [()]
        for depth in range(compressed_count):
            expanded: set[tuple[int, ...]] = set()
            for mask in beam:
                remaining = [layer for layer in order if layer not in mask]
                for layer in remaining:
                    expanded.add(tuple(sorted((*mask, layer))))
            states_examined += len(expanded)
            ranked = sorted(
                expanded,
                key=lambda mask: _allocation_key(proxy, mask, None),
            )
            # Retain a diverse beam. For masks with at least two selected
            # layers, prefer states two swaps apart before filling any slots
            # with closer alternatives.
            next_beam: list[tuple[int, ...]] = []
            seen = set()
            for mask in ranked:
                if mask in seen:
                    continue
                distance = 4 if depth >= 1 else 2
                if all(len(set(mask) ^ set(previous)) >= distance for previous in next_beam):
                    next_beam.append(mask)
                    seen.add(mask)
                    if len(next_beam) == beam_width:
                        break
            if len(next_beam) < beam_width:
                for mask in ranked:
                    if mask not in seen:
                        next_beam.append(mask)
                        seen.add(mask)
                        if len(next_beam) == beam_width:
                            break
            beam = next_beam
        all_masks.update(beam)

    # Improve each beam endpoint with best-improving exact 1-for-1 swaps under
    # the pairwise proxy. The full model is not invoked during this refinement.
    refined = set()
    candidate_set = set(layers)
    for start in all_masks:
        current = start
        while True:
            compressed = set(current)
            dense = candidate_set - compressed
            neighbors = {
                tuple(sorted((compressed - {leave}) | {enter}))
                for leave in compressed for enter in dense
            }
            states_examined += len(neighbors)
            best = min(
                neighbors | {current},
                key=lambda mask: _allocation_key(proxy, mask, None),
            )
            if best == current:
                break
            current = best
        refined.add(current)
    all_masks.update(refined)
    return all_masks, states_examined


def search_fixed_compression_count(
    proxy: FisherAllocationProxy,
    compressed_count: int,
    *,
    max_ppl: float | None = None,
    exact_proxy_limit: int = 256,
    beam_width: int = 16,
    random_seed: int = 1729,
) -> AllocationSearchResult:
    """Select an allocation from the cheap interaction proxy.

    Small candidate sets are enumerated *in proxy space*, which requires no
    additional model forward passes. Larger sets use conditional beam search
    plus proxy-only 1-for-1 refinement. Exact model evaluation of the selected
    allocation remains a separate required validation step.
    """
    n = len(proxy.layers)
    if not 0 <= compressed_count <= n:
        raise ValueError(f"compressed_count must be between 0 and {n}")
    if exact_proxy_limit < 1 or beam_width < 1:
        raise ValueError("exact_proxy_limit and beam_width must be positive")
    combination_count = math.comb(n, compressed_count)
    if combination_count <= exact_proxy_limit:
        masks = combinations(proxy.layers, compressed_count)
        ranked = _rank_allocations(proxy, masks, max_ppl)
        return AllocationSearchResult(
            selected_layers=tuple(ranked[0]["compressed_layers"]),
            ranked_allocations=tuple(ranked),
            strategy="exact_over_fisher_proxy",
            proxy_allocations_scored=len(ranked),
            is_proxy_optimal=True,
        )

    masks, states_examined = _beam_proxy_masks(
        proxy, compressed_count, beam_width=beam_width, random_seed=random_seed,
    )
    ranked = _rank_allocations(proxy, masks, max_ppl)
    if not ranked:
        raise RuntimeError("Proxy search produced no fixed-count allocations")
    return AllocationSearchResult(
        selected_layers=tuple(ranked[0]["compressed_layers"]),
        ranked_allocations=tuple(ranked),
        strategy="conditional_fisher_beam_with_swap_refinement",
        proxy_allocations_scored=states_examined,
        is_proxy_optimal=False,
    )
