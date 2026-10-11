"""Convergence and runtime benchmarking (Reviewer 2, Concern 4; instructions Sec. 6.2).

Convergence criterion (made precise, replacing the original manuscript's
unspecified "95% confidence and standard error below 0.01"): we track
the standard error of the mean (SEM = sample std / sqrt(N)) of P_TR, in
units of P_TR itself (dimensionless probability), and separately its
ratio to the running mean (relative SEM, `relative_sem_pct` below, a
distinct quantity from the SEM itself -- not to be confused with a
confidence-interval half-width). The 95% CI half-width (1.9600 * SEM,
`ci95_halfwidth` below) is also reported, in the same absolute units as
P_TR, for readers who want an absolute rather than relative convergence
indicator. Convergence is declared once the relative SEM falls and stays
below 1%.
"""
from __future__ import annotations

import time

import numpy as np

from .simulation import run_simulation, summarize_ptr


def convergence_table(sizes: list[int], seed: int = 20260716) -> list[dict]:
    rows = []
    for n in sizes:
        t0 = time.perf_counter()
        out = run_simulation(n, seed=seed)
        elapsed = time.perf_counter() - t0
        stats = summarize_ptr(out["P_TR"])
        ci95_halfwidth = 1.959964 * stats["sem"]
        rows.append({
            "N": n,
            "mean_P_TR": stats["mean"],
            "sem": stats["sem"],
            "ci95_halfwidth": ci95_halfwidth,
            "relative_sem_pct": 100.0 * stats["sem"] / stats["mean"],
            "wall_time_s": elapsed,
        })
    return rows


def benchmark_runtime(size: int = 10_000_000, seed: int = 20260716, n_runs: int = 3) -> dict:
    """Honest, reproducible benchmark of the full simulation (sampling + risk-function
    evaluation), matching the manuscript's claimed 10-million-iteration run.
    Does NOT include figure generation or disk I/O (measured separately)."""
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        run_simulation(size, seed=seed + i)
        times.append(time.perf_counter() - t0)
    return {
        "N": size,
        "n_runs": n_runs,
        "times_s": times,
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
    }
