"""Fixed-total projection budgets for cross-layer compression research."""


PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def projection_budgets(mode, r_bridge, n_outliers):
    """Return per-projection budgets while preserving the layerwise total.

    The interaction-aware allocation follows the measured two-layer dense
    restore ordering: Q and O were least helpful, while Gate and Down were the
    strongest individual MLP restores. Bridge rank remains unchanged because
    moving bridge rank across rectangular projections changes bytes unevenly.
    """
    if r_bridge < 0 or n_outliers < 0:
        raise ValueError("Compression budgets must be nonnegative")
    if mode == "uniform":
        sparse = {name: int(n_outliers) for name in PROJECTION_NAMES}
    elif mode == "interaction_aware":
        multipliers = {
            "q_proj": 0.5,
            "k_proj": 1.0,
            "v_proj": 1.0,
            "o_proj": 0.5,
            "gate_proj": 1.5,
            "up_proj": 1.0,
            "down_proj": 1.5,
        }
        sparse = {name: int(round(n_outliers * multipliers[name])) for name in PROJECTION_NAMES}
        target = len(PROJECTION_NAMES) * int(n_outliers)
        sparse["down_proj"] += target - sum(sparse.values())
    else:
        raise ValueError(f"Unknown projection budget mode: {mode}")
    result = {
        name: {"r_bridge": int(r_bridge), "n_outliers": sparse[name]}
        for name in PROJECTION_NAMES
    }
    if sum(item["n_outliers"] for item in result.values()) != len(PROJECTION_NAMES) * n_outliers:
        raise RuntimeError("Projection allocation changed the total sparse budget")
    return result
