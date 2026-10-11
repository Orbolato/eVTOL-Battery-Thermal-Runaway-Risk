"""Core Monte Carlo simulation engine (baseline-independent and dependent variants)."""
from __future__ import annotations

import numpy as np

from . import distributions as dist
from . import dependence as dep
from .risk_functions import compute_ptr, AblationConfig, FULL_MODEL

MASTER_SEED = 20260716  # fixed master seed; see code/README.md for the child-stream policy


def sample_inputs(size: int, rng: np.random.Generator, dependence_scenario: str = "independent") -> dict:
    """Draw one batch of the seven input variables, optionally rank-correlated."""
    raw = {
        "SOC0": dist.sample_soc0(size, rng),
        "N_cycles": dist.sample_ncycles(size, rng),
        "T_bat": dist.sample_tbat(size, rng),
        "T_amb": dist.sample_tamb(size, rng),
        "C_rate": dist.sample_crate(size, rng),
        "R_int0": dist.sample_rint0(size, rng),
        "sigma_V": dist.sample_sigma_v(size, rng),
    }
    if dependence_scenario == "independent":
        return raw
    target = dep.target_correlation_matrix(dependence_scenario)
    return dep.iman_conover(raw, target, rng)


def run_simulation(size: int, seed: int = MASTER_SEED, dependence_scenario: str = "independent",
                    ablation: AblationConfig = FULL_MODEL) -> dict:
    """Run one Monte Carlo batch and return inputs + all computed quantities."""
    rng = np.random.default_rng(seed)
    inputs = sample_inputs(size, rng, dependence_scenario)
    results = compute_ptr(inputs, ablation=ablation)
    out = dict(inputs)
    out.update(results)
    return out


def summarize_ptr(ptr: np.ndarray) -> dict:
    ptr = np.asarray(ptr)
    return {
        "n": int(ptr.size),
        "mean": float(np.mean(ptr)),
        "median": float(np.median(ptr)),
        "std": float(np.std(ptr, ddof=1)),
        "p95": float(np.percentile(ptr, 95)),
        "p99": float(np.percentile(ptr, 99)),
        "p999": float(np.percentile(ptr, 99.9)),
        "max": float(np.max(ptr)),
        "sem": float(np.std(ptr, ddof=1) / np.sqrt(ptr.size)),
    }
