"""Schmidt-spectrum measurement and coefficient accounting for Llama weights."""

from __future__ import annotations

import math

import torch


def llama_projection_modes(d_out: int, d_in: int):
    """Return the exact four-site dimensions used by the original NEC code."""
    if (d_out, d_in) == (4096, 4096):
        return (8, 8, 8, 8), (8, 8, 8, 8)
    if (d_out, d_in) == (11008, 4096):
        return (43, 4, 8, 8), (8, 8, 8, 8)
    if (d_out, d_in) == (4096, 11008):
        return (8, 8, 8, 8), (43, 4, 8, 8)
    raise ValueError(f"no audited Llama-2-7B mode factorization for {(d_out, d_in)}")


def tt_site_unfoldings(
    weight: torch.Tensor,
    output_modes: tuple[int, int, int, int] | None = None,
    input_modes: tuple[int, int, int, int] | None = None,
    site_order: tuple[int, int, int, int] = (0, 1, 2, 3),
):
    """Build the three exact MPO cut matrices for a coordinate tensorization."""
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating-point matrix")
    if output_modes is None or input_modes is None:
        output_modes, input_modes = llama_projection_modes(*weight.shape)
    output_modes = tuple(map(int, output_modes))
    input_modes = tuple(map(int, input_modes))
    order = tuple(map(int, site_order))
    if sorted(order) != [0, 1, 2, 3]:
        raise ValueError("site_order must be a permutation of (0,1,2,3)")
    if math.prod(output_modes) != weight.shape[0] or math.prod(input_modes) != weight.shape[1]:
        raise ValueError("mode products do not match weight shape")

    tensor = weight.reshape(*output_modes, *input_modes)
    permuted_axes = tuple(order) + tuple(4 + axis for axis in order)
    tensor = tensor.permute(*permuted_axes)
    ordered_out = tuple(output_modes[axis] for axis in order)
    ordered_in = tuple(input_modes[axis] for axis in order)
    physical = tuple(o * i for o, i in zip(ordered_out, ordered_in))
    paired = tensor.reshape(*ordered_out, *ordered_in).permute(0, 4, 1, 5, 2, 6, 3, 7)
    paired = paired.contiguous().reshape(*physical)
    cuts = []
    for cut in (1, 2, 3):
        left = math.prod(physical[:cut])
        right = math.prod(physical[cut:])
        cuts.append(paired.reshape(left, right))
    return cuts, physical


def two_site_unfolding(
    weight: torch.Tensor,
    output_modes: tuple[int, int],
    input_modes: tuple[int, int],
    axes: tuple[int, int, int, int],
):
    """Return one two-site operator-Schmidt unfolding under a chosen cut."""
    dims = (*output_modes, *input_modes)
    if math.prod(output_modes) != weight.shape[0] or math.prod(input_modes) != weight.shape[1]:
        raise ValueError("two-site factors do not match the matrix dimensions")
    if sorted(axes) != [0, 1, 2, 3]:
        raise ValueError("axes must be a permutation of four tensor modes")
    rows = dims[axes[0]] * dims[axes[1]]
    columns = dims[axes[2]] * dims[axes[3]]
    return weight.reshape(*dims).permute(*axes).contiguous().reshape(rows, columns)


@torch.no_grad()
def singular_spectrum(
    matrix: torch.Tensor,
    *,
    exact: bool = True,
    seed: int = 0,
    max_rank: int = 2048,
    oversample: int = 64,
    power_iterations: int = 2,
):
    """Compute a descending spectrum; ``exact=False`` returns a randomized sketch.

    The Kaggle diagnostic uses exact FP64 Gram eigenvalues only when the
    smaller unfolding dimension is at most 256. Larger cuts use the
    randomized range-SVD path and report captured energy and unresolved
    energy against the exact Frobenius norm.
    """
    original = matrix.detach()
    if original.ndim != 2 or not torch.isfinite(original).all():
        raise ValueError("spectrum input must be a finite rank-two tensor")
    norm_sq = original.to(dtype=torch.float64).square().sum()
    if exact or min(original.shape) <= 256:
        exact_source = original.to(dtype=torch.float64)
        if exact_source.shape[0] <= exact_source.shape[1]:
            gram = exact_source @ exact_source.T
        else:
            gram = exact_source.T @ exact_source
        eigenvalues = torch.linalg.eigvalsh(0.5 * (gram + gram.T)).clamp_min(0)
        values = eigenvalues.sqrt().flip(0)
        return {
            "singular_values": values,
            "frobenius_norm_squared": float(norm_sq),
            "method": "exact_gram_eigenvalues_fp64",
            "estimated": False,
            "captured_energy_fraction": 1.0,
        }

    source = original.to(dtype=torch.float32)
    rows, columns = source.shape
    rank_cap = min(int(max_rank), rows, columns)
    sketch_rank = min(rank_cap + int(oversample), rows, columns)
    if rank_cap <= 0 or oversample < 0 or power_iterations < 0:
        raise ValueError("randomized spectrum rank and iteration settings must be nonnegative")
    generator = torch.Generator(device=source.device)
    generator.manual_seed(int(seed))
    if rows >= columns:
        omega = torch.randn(columns, sketch_rank, generator=generator, device=source.device, dtype=source.dtype)
        sample = source @ omega
        q = torch.linalg.qr(sample, mode="reduced").Q
        for _ in range(int(power_iterations)):
            q = torch.linalg.qr(source @ (source.T @ q), mode="reduced").Q
        small = q.T @ source
        values = torch.linalg.svdvals(small)
    else:
        omega = torch.randn(rows, sketch_rank, generator=generator, device=source.device, dtype=source.dtype)
        sample = source.T @ omega
        q = torch.linalg.qr(sample, mode="reduced").Q
        for _ in range(int(power_iterations)):
            q = torch.linalg.qr(source.T @ (source @ q), mode="reduced").Q
        small = q.T @ source.T
        values = torch.linalg.svdvals(small)
    values = values.sort(descending=True).values[:rank_cap].to(torch.float64)
    captured = values.square().sum() / norm_sq.clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "singular_values": values,
        "frobenius_norm_squared": float(norm_sq),
        "method": "randomized_range_svd_fp32_ritz_values",
        "estimated": True,
        "seed": int(seed),
        "max_rank": int(rank_cap),
        "sketch_rank": int(sketch_rank),
        "power_iterations": int(power_iterations),
        "captured_energy_fraction": float(captured),
        "unresolved_energy_fraction": float((1 - captured).clamp_min(0)),
    }


def tail_energy_fraction(singular_values: torch.Tensor, rank: int, total_sq: float | None = None):
    values = torch.as_tensor(singular_values, dtype=torch.float64)
    total = float(values.square().sum()) if total_sq is None else float(total_sq)
    if total <= 0:
        return 0.0
    rank = max(0, min(int(rank), values.numel()))
    kept = values[:rank].square().sum()
    return max(0.0, (total - float(kept)) / total)


def minimal_rank_for_tail(
    singular_values: torch.Tensor,
    residual_energy_fraction: float,
    total_sq: float | None = None,
):
    """Smallest rank whose discarded squared-energy fraction meets a target."""
    target = float(residual_energy_fraction)
    if not 0 <= target <= 1:
        raise ValueError("residual_energy_fraction must lie in [0,1]")
    values = torch.as_tensor(singular_values, dtype=torch.float64).sort(descending=True).values
    energy = values.square()
    # Randomized spectra may omit a substantial part of the singular spectrum.
    # Callers can pass the exact Frobenius norm so that the omitted energy is
    # conservatively retained in the residual instead of silently renormalized.
    total = energy.sum() if total_sq is None else torch.as_tensor(
        total_sq, dtype=torch.float64, device=energy.device,
    )
    if total <= 0:
        return 0
    threshold = target * total
    tail = total - energy.cumsum(0)
    indices = torch.nonzero(tail <= threshold, as_tuple=False)
    return int(indices[0, 0] + 1) if indices.numel() else int(values.numel())


def tt_parameter_count(physical_dimensions, ranks):
    """Count every scalar in a 4-site TT/MPO core sequence."""
    dims = tuple(map(int, physical_dimensions))
    ranks = tuple(map(int, ranks))
    if len(dims) != 4 or len(ranks) != 5 or ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("expected four physical dimensions and boundary ranks one")
    if any(size <= 0 for size in dims) or any(rank <= 0 for rank in ranks):
        raise ValueError("physical dimensions and ranks must be positive")
    return sum(ranks[i] * dims[i] * ranks[i + 1] for i in range(4))


def tt_rank_profile(
    spectra,
    physical_dimensions,
    error_target: float,
    total_squares: list[float] | tuple[float, ...] | None = None,
):
    """Select TT ranks using the supplied singular spectra and Frobenius totals.

    The total Frobenius-error target allocates ``error_target**2 / 3`` to each
    of the three cuts.  When ``total_squares`` is supplied, omitted energy in
    a randomized spectrum counts toward the residual.  With exact spectra the
    computed error is the usual TT-SVD bound. With randomized spectra it is
    only an estimate based on the supplied Ritz values, not a certificate or
    a globally minimal integer rank-vector search.
    """
    if len(spectra) != 3 or len(physical_dimensions) != 4:
        raise ValueError("a four-site TT needs three cut spectra and four dimensions")
    if total_squares is not None and len(total_squares) != 3:
        raise ValueError("total_squares must contain one exact Frobenius total per cut")
    error_target = float(error_target)
    if not 0 < error_target < 1:
        raise ValueError("error_target must lie strictly between zero and one")
    local_tail = error_target**2 / 3.0
    totals = [None] * 3 if total_squares is None else list(total_squares)
    ranks = [1] + [
        minimal_rank_for_tail(spectrum, local_tail, total_sq=total)
        for spectrum, total in zip(spectra, totals)
    ] + [1]
    # Enforce r[k+1] <= r[k] d[k] and r[k] <= d[k+1] r[k+1].
    changed = True
    while changed:
        changed = False
        for k in range(1, 4):
            needed = math.ceil(ranks[k] / physical_dimensions[k])
            if ranks[k + 1] < needed:
                ranks[k + 1] = needed
                changed = True
        for k in range(3, 0, -1):
            needed = math.ceil(ranks[k + 1] / physical_dimensions[k])
            if ranks[k] < needed:
                ranks[k] = needed
                changed = True
    if any(ranks[k] > min(math.prod(physical_dimensions[:k]), math.prod(physical_dimensions[k:])) for k in range(1, 4)):
        raise RuntimeError("selected TT ranks exceed exact unfolding dimensions")
    tails = [
        tail_energy_fraction(s, ranks[k + 1], total_sq=totals[k])
        for k, s in enumerate(spectra)
    ]
    return {
        "ranks": ranks,
        "coefficient_count": tt_parameter_count(physical_dimensions, ranks),
        "cut_residual_energy_fraction": tails,
        "tt_error_bound": math.sqrt(sum(tails)),
        "requested_error_bound": error_target,
    }


def spectra_summary(
    spectrum: dict,
    residual_targets=(0.01, 0.05, 0.10, 0.20),
    retained_rank: int | None = None,
):
    """Return JSON-safe energy and entropy diagnostics for one Schmidt cut.

    Randomized spectra do not contain the unresolved tail.  Their reported
    partial entropy uses the exact Frobenius energy for normalization but is
    not labeled as the entropy of the unknown full spectrum.  Exact spectra
    report the true operator-Schmidt entropy.
    """
    values = spectrum["singular_values"]
    total = spectrum["frobenius_norm_squared"]
    rank_records = {}
    for target in residual_targets:
        rank = minimal_rank_for_tail(values, target, total_sq=total)
        cap_tail = tail_energy_fraction(values, values.numel(), total)
        reached = cap_tail <= float(target)
        rank_records[str(target)] = {
            "minimum_rank": rank if reached else None,
            "rank_lower_bound": rank if reached else int(values.numel()),
            "residual_energy_at_reported_rank": (
                tail_energy_fraction(values, rank, total) if reached else cap_tail
            ),
            "target_reached_with_spectrum": reached,
        }
    probabilities = values.double().square() / max(float(total), torch.finfo(torch.float64).tiny)
    nonzero = probabilities > 0
    entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    result = {
        "method": spectrum["method"],
        "estimated": spectrum["estimated"],
        "singular_value_count": int(values.numel()),
        "captured_energy_fraction": float(spectrum["captured_energy_fraction"]),
        "schmidt_entropy_nats": entropy if not spectrum["estimated"] else None,
        "reported_spectrum_partial_entropy_nats": entropy,
        "effective_schmidt_rank": math.exp(entropy) if not spectrum["estimated"] else None,
        "reported_spectrum_partial_entropy_exp": math.exp(entropy),
        "tail_energy_by_rank": {
            str(rank): tail_energy_fraction(values, rank, total)
            for rank in sorted({1, min(8, values.numel()), min(64, values.numel()), min(256, values.numel()), values.numel()})
        },
        "min_rank_by_residual_energy_fraction": rank_records,
    }
    if retained_rank is not None:
        result["configured_retained_rank"] = int(retained_rank)
        result["configured_rank_residual_energy_fraction"] = tail_energy_fraction(
            values, retained_rank, total,
        )
    return result


@torch.no_grad()
def named_tt_spectrum_audit(
    matrices: dict[str, torch.Tensor],
    output_modes: tuple[int, int, int, int],
    input_modes: tuple[int, int, int, int],
    ranks: tuple[int, int, int, int, int],
    *,
    site_order: tuple[int, int, int, int] = (0, 1, 2, 3),
    exact_large_cuts: bool = False,
    seed: int = 0,
    max_rank: int = 512,
    oversample: int = 32,
    power_iterations: int = 2,
):
    """Measure matched operator-Schmidt spectra for named matrix variants.

    The same modes, site order, ranks, and numerical method are used for every
    variant, making before/after comparisons meaningful.  Cuts with a smaller
    dimension of at most 256 are always exact; larger cuts are randomized
    unless ``exact_large_cuts`` is requested.  Randomized reports retain the
    unresolved Frobenius energy instead of renormalizing the observed Ritz
    values.
    """
    if not matrices:
        raise ValueError("at least one named matrix is required")
    ranks = tuple(map(int, ranks))
    if len(ranks) != 5 or ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("ranks must contain five TT ranks with unit boundaries")
    expected_shape = (math.prod(output_modes), math.prod(input_modes))
    report = {}
    reference_physical = None
    for name, matrix in matrices.items():
        if tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"matrix {name!r} has shape {tuple(matrix.shape)}, expected {expected_shape}"
            )
        cuts, physical = tt_site_unfoldings(
            matrix, output_modes, input_modes, site_order,
        )
        if reference_physical is None:
            reference_physical = physical
        elif physical != reference_physical:
            raise RuntimeError("matched spectrum variants produced different physical dimensions")
        cut_records = []
        for cut_index, unfolding in enumerate(cuts):
            spectrum = singular_spectrum(
                unfolding,
                exact=bool(exact_large_cuts),
                # Use the same random test subspace seed for every named
                # variant at a given cut.  This makes before/after deltas a
                # matched numerical comparison instead of adding sketch noise.
                seed=int(seed) + cut_index,
                max_rank=max_rank,
                oversample=oversample,
                power_iterations=power_iterations,
            )
            summary = spectra_summary(spectrum, retained_rank=ranks[cut_index + 1])
            summary.update({
                "cut_after_site": cut_index,
                "unfolding_shape": [int(value) for value in unfolding.shape],
            })
            cut_records.append(summary)
        report[name] = cut_records
    return {
        "output_modes": list(map(int, output_modes)),
        "input_modes": list(map(int, input_modes)),
        "site_order": list(map(int, site_order)),
        "physical_dimensions": list(map(int, reference_physical)),
        "configured_ranks": list(ranks),
        "exact_large_cuts": bool(exact_large_cuts),
        "randomized_settings": {
            "seed": int(seed),
            "max_rank": int(max_rank),
            "oversample": int(oversample),
            "power_iterations": int(power_iterations),
        },
        "variants": report,
    }
