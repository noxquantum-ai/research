"""Byte-exact planning for NEC's adaptive binary Schmidt representation.

This planner is architecture neutral.  It assigns bond ranks to arbitrary
matrix operators under a serialized byte budget; later calibration can move
rank between operators according to measured function loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from nox.model_adapter import ModelInventory


@dataclass(frozen=True)
class SchmidtTargetPlan:
    name: str
    shape: tuple[int, int]
    rank: int
    dense_bytes: int
    packed_sign_bytes: int
    scale_bytes: int
    bias_bytes: int
    payload_bytes: int
    effective_bits_per_weight: float

    def to_dict(self) -> dict:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True)
class SchmidtModelPlan:
    requested_ratio: float
    achieved_payload_ratio: float
    dense_model_bytes: int
    compressed_payload_bytes: int
    untargeted_bytes: int
    target_plans: tuple[SchmidtTargetPlan, ...]
    feasible: bool
    infeasibility_reason: str | None

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "target_plans": [target.to_dict() for target in self.target_plans],
        }


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def schmidt_payload_bytes(
    out_features: int,
    in_features: int,
    rank: int,
    *,
    scale_bits: int = 16,
) -> tuple[int, int]:
    """Return packed sign and scale bytes for one binary Schmidt operator.

    The operator stores two one-bit factor matrices, output/input magnitude
    vectors, and one magnitude per bond coordinate.  Byte padding is charged
    separately for each packed factor so the estimate matches serialization.
    """

    if min(out_features, in_features, rank) <= 0:
        raise ValueError("operator dimensions and rank must be positive")
    if scale_bits not in (16, 32):
        raise ValueError("scale_bits must be 16 or 32")
    sign_bytes = _ceil_div(out_features * rank, 8) + _ceil_div(rank * in_features, 8)
    scale_bytes = _ceil_div(scale_bits * (out_features + in_features + rank), 8)
    return sign_bytes, scale_bytes


def _rank_for_budget(out_features: int, in_features: int, budget: int, scale_bits: int) -> int:
    fixed_scale_bytes = _ceil_div(scale_bits * (out_features + in_features), 8)
    if budget <= fixed_scale_bytes:
        return 0
    # Find the largest byte-feasible rank.  Binary search handles per-factor
    # byte padding exactly and is cheap relative to model loading.
    low, high = 0, min(out_features, in_features)
    while low < high:
        mid = (low + high + 1) // 2
        signs, scales = schmidt_payload_bytes(
            out_features, in_features, mid, scale_bits=scale_bits,
        )
        if signs + scales <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def plan_uniform_schmidt_budget(
    inventory: ModelInventory,
    *,
    target_ratio: float,
    scale_bits: int = 16,
    minimum_rank: int = 8,
    metadata_reserve_bytes: int = 1 << 20,
) -> SchmidtModelPlan:
    """Create a first-pass rank plan under a whole-model byte constraint.

    Shared input/output embeddings are charged once.  Biases and parameters
    outside matrix modules remain dense.  The metadata reserve prevents a
    mathematically exact payload target from overflowing after packaging.
    """

    if target_ratio <= 1:
        raise ValueError("target_ratio must exceed one")
    if minimum_rank <= 0:
        raise ValueError("minimum_rank must be positive")
    dense_bytes = inventory.total_unique_parameter_bytes
    package_budget = math.floor(dense_bytes / target_ratio) - metadata_reserve_bytes
    compressible_budget = package_budget - inventory.untargeted_parameter_bytes

    unique_targets = []
    seen_groups: set[int] = set()
    for target in inventory.targets:
        if target.tied_group in seen_groups:
            continue
        seen_groups.add(target.tied_group)
        unique_targets.append(target)
    dense_target_bytes = sum(target.weight_bytes for target in unique_targets)
    dense_bias_bytes = sum(target.bias_bytes for target in inventory.targets)
    factor_budget = compressible_budget - dense_bias_bytes

    if factor_budget <= 0 or dense_target_bytes <= 0:
        return SchmidtModelPlan(
            requested_ratio=target_ratio,
            achieved_payload_ratio=1.0,
            dense_model_bytes=dense_bytes,
            compressed_payload_bytes=dense_bytes,
            untargeted_bytes=inventory.untargeted_parameter_bytes,
            target_plans=(),
            feasible=False,
            infeasibility_reason="unchanged parameters and packaging consume the requested budget",
        )

    plans: list[SchmidtTargetPlan] = []
    payload_total = inventory.untargeted_parameter_bytes + dense_bias_bytes + metadata_reserve_bytes
    for target in unique_targets:
        share = factor_budget * target.weight_bytes // dense_target_bytes
        out_features, in_features = target.shape
        rank = _rank_for_budget(out_features, in_features, share, scale_bits)
        if rank < minimum_rank:
            return SchmidtModelPlan(
                requested_ratio=target_ratio,
                achieved_payload_ratio=dense_bytes / max(payload_total, 1),
                dense_model_bytes=dense_bytes,
                compressed_payload_bytes=payload_total,
                untargeted_bytes=inventory.untargeted_parameter_bytes,
                target_plans=tuple(plans),
                feasible=False,
                infeasibility_reason=f"{target.name} receives rank {rank}, below minimum {minimum_rank}",
            )
        sign_bytes, scale_bytes = schmidt_payload_bytes(
            out_features, in_features, rank, scale_bits=scale_bits,
        )
        payload = sign_bytes + scale_bytes + target.bias_bytes
        payload_total += sign_bytes + scale_bytes
        plans.append(SchmidtTargetPlan(
            name=target.name,
            shape=target.shape,
            rank=rank,
            dense_bytes=target.weight_bytes + target.bias_bytes,
            packed_sign_bytes=sign_bytes,
            scale_bytes=scale_bytes,
            bias_bytes=target.bias_bytes,
            payload_bytes=payload,
            effective_bits_per_weight=(8 * (sign_bytes + scale_bytes)) / target.scalar_count,
        ))

    ratio = dense_bytes / payload_total
    feasible = payload_total <= package_budget + metadata_reserve_bytes
    return SchmidtModelPlan(
        requested_ratio=target_ratio,
        achieved_payload_ratio=ratio,
        dense_model_bytes=dense_bytes,
        compressed_payload_bytes=payload_total,
        untargeted_bytes=inventory.untargeted_parameter_bytes,
        target_plans=tuple(plans),
        feasible=feasible,
        infeasibility_reason=None if feasible else "rank rounding exceeded the requested package budget",
    )
