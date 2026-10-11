"""Mitigation strategies with operational trade-off indices (R1.7, R2.9).

The original manuscript evaluated SOC/C-rate/cycle-replacement limits
purely as risk reductions via truncated resampling, with no accounting
for the operational cost of those restrictions. Here every strategy is
paired with transparent, explicitly labeled proxy indices for
turnaround time, usable energy window, and replacement frequency, so
that risk reduction is never presented without its operational cost.
These are declared *proxy* indices (not validated economic or
operational models) and are reported as relative, dimensionless ratios
to the unrestricted baseline.
"""
from __future__ import annotations

import numpy as np

from .simulation import sample_inputs, MASTER_SEED
from .risk_functions import compute_ptr, FULL_MODEL

SOC_FLOOR = 30.0  # assumed minimum practical SOC at charge initiation (Table I support)
UNRESTRICTED_N_CAP = 5000.0  # the Table I engineering cap on N_cycles; used as the
                               # "unrestricted baseline" replacement threshold so that
                               # replacement_rate_index(None) == 1.00, consistent with
                               # every other proxy index in this module being defined
                               # relative to the unrestricted baseline.


def _truncated_resample(size: int, rng: np.random.Generator, soc_max=None, c_max=None, n_max=None) -> dict:
    """Draw inputs, resampling SOC0/C_rate/N_cycles until each falls within the
    mitigated operational envelope (rejection sampling on the corresponding marginal
    only, exactly as in the original manuscript's approach, so the risk-reduction
    numbers remain comparable to the baseline formulation)."""
    inputs = sample_inputs(size, rng, dependence_scenario="independent")
    if soc_max is not None:
        mask = inputs["SOC0"] > soc_max
        while mask.any():
            resampled = sample_inputs(mask.sum(), rng, dependence_scenario="independent")
            inputs["SOC0"][mask] = resampled["SOC0"]
            mask = inputs["SOC0"] > soc_max
    if c_max is not None:
        mask = inputs["C_rate"] > c_max
        while mask.any():
            resampled = sample_inputs(mask.sum(), rng, dependence_scenario="independent")
            inputs["C_rate"][mask] = resampled["C_rate"]
            mask = inputs["C_rate"] > c_max
    if n_max is not None:
        mask = inputs["N_cycles"] > n_max
        while mask.any():
            resampled = sample_inputs(mask.sum(), rng, dependence_scenario="independent")
            inputs["N_cycles"][mask] = resampled["N_cycles"]
            mask = inputs["N_cycles"] > n_max
    return inputs


def turnaround_time_index(c_rate_max: float | None, soc_max: float, soc_floor: float = SOC_FLOOR) -> float:
    """Proxy charging-time index relative to the unrestricted baseline (SOC_max=95,
    C_max=6C), assuming charge time is proportional to (SOC_max - SOC_floor)/C_rate_max
    (constant-current-equivalent approximation). Larger index = longer turnaround."""
    c_eff = 6.0 if c_rate_max is None else c_rate_max
    baseline_time = (95.0 - soc_floor) / 6.0
    time = (soc_max - soc_floor) / c_eff
    return time / baseline_time


def usable_energy_index(soc_max: float, soc_floor: float = SOC_FLOOR) -> float:
    """Usable SOC window relative to the unrestricted baseline (95 - 30 = 65 pp)."""
    return (soc_max - soc_floor) / (95.0 - soc_floor)


def replacement_rate_index(n_max: float | None) -> float:
    """Battery replacements per unit operational time relative to the unrestricted
    baseline (replace only at the UNRESTRICTED_N_CAP=5000-cycle engineering cap, so
    replacement_rate_index(None) == 1.00, matching turnaround_time_index and
    usable_energy_index's normalization convention). Higher index = more frequent,
    costlier replacements."""
    n_eff = UNRESTRICTED_N_CAP if n_max is None else n_max
    return UNRESTRICTED_N_CAP / n_eff


def evaluate_strategy(size: int, seed: int, soc_max=None, c_max=None, n_max=None) -> dict:
    rng = np.random.default_rng(seed)
    inputs = _truncated_resample(size, rng, soc_max=soc_max, c_max=c_max, n_max=n_max)
    ptr = compute_ptr(inputs, ablation=FULL_MODEL)["P_TR"]
    return {
        "mean_P_TR": float(np.mean(ptr)),
        "turnaround_time_index": turnaround_time_index(c_max, soc_max if soc_max else 95.0),
        "usable_energy_index": usable_energy_index(soc_max if soc_max else 95.0),
        "replacement_rate_index": replacement_rate_index(n_max),
    }


STRATEGIES = {
    "baseline": dict(soc_max=None, c_max=None, n_max=None),
    "soc_85": dict(soc_max=85.0, c_max=None, n_max=None),
    "soc_80": dict(soc_max=80.0, c_max=None, n_max=None),
    "crate_5": dict(soc_max=None, c_max=5.0, n_max=None),
    "crate_4": dict(soc_max=None, c_max=4.0, n_max=None),
    "cycles_1200": dict(soc_max=None, c_max=None, n_max=1200.0),
    "combined_moderate": dict(soc_max=85.0, c_max=4.0, n_max=1200.0),
    "combined_aggressive": dict(soc_max=80.0, c_max=3.5, n_max=1000.0),
}


def run_all_strategies(size: int = 2_000_000, seed: int = MASTER_SEED) -> list[dict]:
    baseline = evaluate_strategy(size, seed, **STRATEGIES["baseline"])
    rows = []
    for name, params in STRATEGIES.items():
        r = evaluate_strategy(size, seed, **params)
        r["strategy"] = name
        r["risk_reduction_pct"] = 100.0 * (baseline["mean_P_TR"] - r["mean_P_TR"]) / baseline["mean_P_TR"]
        rows.append(r)
    return rows
