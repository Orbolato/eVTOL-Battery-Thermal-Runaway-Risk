#!/usr/bin/env python
"""Single entry point regenerating every table and figure quoted in the revised
manuscript and response to reviewers. Run from the code/ directory:

    python run_all.py

All outputs are written under outputs/tables, outputs/figures, and outputs/logs.
A result_manifest.csv records, for every reported number, which output file and
which script produced it, so every number in the manuscript is traceable.
"""
from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from scipy import stats as scipy_stats

from src.simulation import run_simulation, summarize_ptr, MASTER_SEED
from src import sensitivity, mitigation, convergence as conv, plotting
from src.risk_functions import (
    ABLATION_SCENARIOS, FULL_MODEL, P_BASE_ASSUMED, compute_ptr,
)
from src.dependence import target_correlation_matrix
from src.simulation import sample_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "outputs", "tables")
FIGURES = os.path.join(HERE, "outputs", "figures")
LOGS = os.path.join(HERE, "outputs", "logs")
for d in (TABLES, FIGURES, LOGS):
    os.makedirs(d, exist_ok=True)

N_MAIN = 10_000_000
N_SOBOL = 1_000_000
N_MITIGATION = 2_000_000
N_DEPENDENCE = 2_000_000
N_BENCHMARK_RUNS = 5  # fixed at 5 everywhere this benchmark is invoked/reported, so a
                        # full-pipeline rerun can never silently overwrite the stored
                        # runtime_benchmark.json with a different run count than the
                        # manuscript quotes.

TOP_VARS = ["N_cycles", "SOC0"]  # variables whose ranking robustness is checked across
                                   # every dependence/ablation scenario (Sec. IV-B/IV-C)


def quick_rank_check(out: dict) -> dict:
    """Point-estimate Spearman rho for the two variables the manuscript claims remain
    dominant "across scenarios" (Sec. V-A); no bootstrap CI (this is a cheap ranking
    check reusing samples already drawn for the scenario, not a full sensitivity
    analysis in its own right -- see sensitivity.spearman_bootstrap for that, computed
    only on the independent baseline per Sec. III-I)."""
    ptr = out["P_TR"]
    return {v: float(scipy_stats.spearmanr(out[v], ptr)[0]) for v in TOP_VARS}

manifest_rows = []
log_lines = []


def log(msg: str) -> None:
    print(msg)
    log_lines.append(msg)


def record(quantity, value, source_file, script_section):
    manifest_rows.append({
        "quantity": quantity,
        "value": value,
        "output_file": source_file,
        "produced_by": script_section,
    })


def main():
    t_start = time.time()
    log(f"run_all.py started {time.ctime()}")
    log(f"Platform: {platform.platform()}")
    log(f"Python: {sys.version}")
    log(f"NumPy: {np.__version__}")

    # ---------------------------------------------------------------
    # 1. Baseline (independent) model, N = 10,000,000
    # ---------------------------------------------------------------
    log("\n[1] Baseline independent-model simulation (N=%d)" % N_MAIN)
    t0 = time.time()
    baseline = run_simulation(N_MAIN, seed=MASTER_SEED, dependence_scenario="independent", ablation=FULL_MODEL)
    baseline_summary = summarize_ptr(baseline["P_TR"])
    baseline_time = time.time() - t0
    log(f"  mean P_TR = {baseline_summary['mean']:.4e}  (elapsed {baseline_time:.2f} s)")

    with open(os.path.join(TABLES, "baseline_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baseline_summary.keys()))
        w.writeheader()
        w.writerow(baseline_summary)
    record("baseline_mean_P_TR", baseline_summary["mean"], "baseline_summary.csv", "run_all.py:1")

    fold_vs_base = baseline_summary["mean"] / P_BASE_ASSUMED
    log(f"  ratio to assumed automotive reference P_base={P_BASE_ASSUMED:.2e}: {fold_vs_base:.2f}x")

    # Purely nominal magnitude bands (Band I-IV), not evaluative labels: earlier
    # drafts called these "Acceptable/Low/Medium/Critical," which reads as a
    # regulatory/safety judgment the model cannot support. Renamed per the
    # resubmission audit; see access_revised.tex Sec. IV-A.
    ptr = baseline["P_TR"]
    levels = {
        "Band I (<1e-9)": np.mean(ptr < 1e-9),
        "Band II (1e-9 to Pbase)": np.mean((ptr >= 1e-9) & (ptr < P_BASE_ASSUMED)),
        "Band III (Pbase to 1e-6)": np.mean((ptr >= P_BASE_ASSUMED) & (ptr < 1e-6)),
        "Band IV (>=1e-6)": np.mean(ptr >= 1e-6),
    }
    with open(os.path.join(TABLES, "risk_levels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["risk_level", "fraction"])
        for k, v in levels.items():
            w.writerow([k, v])
    log(f"  risk level fractions: {levels}")

    plotting.fig_distribution(ptr, P_BASE_ASSUMED, os.path.join(FIGURES, "fig2_ptr_distribution.pdf"))

    # ---------------------------------------------------------------
    # 2. Structural ablation matrix (double-counting quantification)
    # ---------------------------------------------------------------
    log("\n[2] Structural ablation matrix (N=%d each)" % N_MITIGATION)
    ablation_rows = []
    for ab in ABLATION_SCENARIOS:
        out = run_simulation(N_MITIGATION, seed=MASTER_SEED, dependence_scenario="independent", ablation=ab)
        s = summarize_ptr(out["P_TR"])
        rank = quick_rank_check(out)
        ablation_rows.append({
            "label": ab.label, "mean_P_TR": s["mean"], "p95_P_TR": s["p95"],
            "rho_N_cycles": rank["N_cycles"], "rho_SOC0": rank["SOC0"],
        })
        log(f"  {ab.label}: mean={s['mean']:.4e}  rho(N_cycles)={rank['N_cycles']:.3f}  rho(SOC0)={rank['SOC0']:.3f}")
    with open(os.path.join(TABLES, "ablation_matrix.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ablation_rows[0].keys()))
        w.writeheader()
        w.writerows(ablation_rows)

    plotting.fig_dependency_graph(os.path.join(FIGURES, "fig1_dependency_graph.pdf"))

    # ---------------------------------------------------------------
    # 3. Dependence scenarios (Iman-Conover induced rank correlation)
    # ---------------------------------------------------------------
    log("\n[3] Dependence scenarios (N=%d each)" % N_DEPENDENCE)
    dependence_rows = []
    for scenario in ["independent", "low", "base", "high", "adverse"]:
        out = run_simulation(N_DEPENDENCE, seed=MASTER_SEED, dependence_scenario=scenario, ablation=FULL_MODEL)
        s = summarize_ptr(out["P_TR"])
        rank = quick_rank_check(out)
        dependence_rows.append({
            "scenario": scenario, "mean_P_TR": s["mean"], "p95_P_TR": s["p95"],
            "rho_N_cycles": rank["N_cycles"], "rho_SOC0": rank["SOC0"],
        })
        log(f"  {scenario}: mean={s['mean']:.4e}  rho(N_cycles)={rank['N_cycles']:.3f}  rho(SOC0)={rank['SOC0']:.3f}")
    with open(os.path.join(TABLES, "dependence_scenarios.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dependence_rows[0].keys()))
        w.writeheader()
        w.writerows(dependence_rows)

    # ---------------------------------------------------------------
    # 4. Convergence and runtime benchmark
    # ---------------------------------------------------------------
    log("\n[4] Convergence experiment")
    conv_rows = conv.convergence_table([10_000, 100_000, 1_000_000, 10_000_000])
    with open(os.path.join(TABLES, "convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conv_rows[0].keys()))
        w.writeheader()
        w.writerows(conv_rows)
    for r in conv_rows:
        log(f"  N={r['N']:>10d}  mean={r['mean_P_TR']:.4e}  rel.SEM={r['relative_sem_pct']:.3f}%  t={r['wall_time_s']:.2f}s")
    plotting.fig_convergence(conv_rows, os.path.join(FIGURES, "fig_convergence.pdf"))

    log(f"\n[4b] Runtime benchmark (N=10,000,000, {N_BENCHMARK_RUNS} runs)")
    bench = conv.benchmark_runtime(size=10_000_000, n_runs=N_BENCHMARK_RUNS)
    with open(os.path.join(TABLES, "runtime_benchmark.json"), "w") as f:
        json.dump({
            "hardware": "Intel Core i7-8565U CPU @ 1.80GHz, 4 cores/8 threads, 7.88 GB RAM",
            "os": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "note": "single-threaded NumPy vectorized simulation; sampling + risk-function "
                    "evaluation only, excludes figure generation and disk I/O",
            **bench,
        }, f, indent=2)
    log(f"  median runtime over {bench['n_runs']} runs: {bench['median_s']:.2f} s "
        f"(range {bench['min_s']:.2f}-{bench['max_s']:.2f} s)")

    # ---------------------------------------------------------------
    # 5. Sensitivity: Spearman with bootstrap CIs + Sobol indices
    # ---------------------------------------------------------------
    log("\n[5] Sensitivity analysis")
    spearman_rows = sensitivity.spearman_bootstrap(baseline, baseline["P_TR"], n_boot=300)
    with open(os.path.join(TABLES, "sensitivity_spearman.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(spearman_rows[0].keys()))
        w.writeheader()
        w.writerows(spearman_rows)
    for r in spearman_rows:
        log(f"  {r['variable']}: rho_s={r['rho_s']:.4f}  95% CI=({r['ci95_lo']:.4f}, {r['ci95_hi']:.4f})")
    plotting.fig_sensitivity(spearman_rows, os.path.join(FIGURES, "fig3_sensitivity.pdf"))

    sobol_rows = sensitivity.sobol_indices(size=N_SOBOL)
    with open(os.path.join(TABLES, "sensitivity_sobol.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sobol_rows[0].keys()))
        w.writeheader()
        w.writerows(sobol_rows)
    for r in sobol_rows:
        log(f"  {r['variable']}: S1={r['S1']:.4f}  ST={r['ST']:.4f}")
    plotting.fig_sobol(sobol_rows, os.path.join(FIGURES, "fig_sobol.pdf"))

    # ---------------------------------------------------------------
    # 6. SOC x C-rate heatmap (independent baseline model)
    # ---------------------------------------------------------------
    log("\n[6] SOC x C-rate interaction heatmap")
    n_grid = 40
    soc_edges = np.linspace(30, 95, n_grid + 1)
    crate_edges = np.linspace(3, 6, n_grid + 1)
    rng = np.random.default_rng(MASTER_SEED)
    n_per_cell = 4000
    mean_grid = np.zeros((n_grid, n_grid))
    for i in range(n_grid):
        for j in range(n_grid):
            soc_mid = 0.5 * (soc_edges[i] + soc_edges[i + 1])
            crate_mid = 0.5 * (crate_edges[j] + crate_edges[j + 1])
            inputs = sample_inputs(n_per_cell, rng, dependence_scenario="independent")
            inputs["SOC0"] = np.full(n_per_cell, soc_mid)
            inputs["C_rate"] = np.full(n_per_cell, crate_mid)
            ptr_cell = compute_ptr(inputs, ablation=FULL_MODEL)["P_TR"]
            mean_grid[i, j] = ptr_cell.mean()
    np.savetxt(os.path.join(TABLES, "heatmap_soc_crate_mean_ptr.csv"), mean_grid, delimiter=",")
    plotting.fig_heatmap(soc_edges, crate_edges, mean_grid, P_BASE_ASSUMED, os.path.join(FIGURES, "fig4_heatmap.pdf"))
    log(f"  grid mean_P_TR range: [{mean_grid.min():.3e}, {mean_grid.max():.3e}]")

    # ---------------------------------------------------------------
    # 7. Mitigation strategies + operational trade-offs
    # ---------------------------------------------------------------
    log("\n[7] Mitigation strategies (N=%d each)" % N_MITIGATION)
    mitigation_rows = mitigation.run_all_strategies(size=N_MITIGATION)
    with open(os.path.join(TABLES, "mitigation_strategies.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mitigation_rows[0].keys()))
        w.writeheader()
        w.writerows(mitigation_rows)
    for r in mitigation_rows:
        log(f"  {r['strategy']}: mean_P_TR={r['mean_P_TR']:.4e}  "
            f"risk_reduction={r['risk_reduction_pct']:.1f}%  "
            f"turnaround_idx={r['turnaround_time_index']:.2f}  "
            f"replacement_idx={r['replacement_rate_index']:.2f}")
    plotting.fig_mitigation_pareto(mitigation_rows, os.path.join(FIGURES, "fig5_mitigation_pareto.pdf"))

    # NOTE: an earlier revision of this script also ran a sequential, age-evolving
    # "life-cycle" simulation and a fleet-level extrapolation here. Both are removed:
    # the life-cycle simulation reused the N_cycles marginal (Table I; fit as a
    # cross-sectional "current accumulated cycles at a random charge event"
    # distribution) as if it were instead each pack's forward-looking replacement-
    # threshold/lifetime distribution. These are not the same random variable, and
    # conflating them undermines both the life-cycle result and, by extension, the
    # baseline cycle-count sensitivity result Reviewer 1 specifically challenged. See
    # baseline_audit.md, Sec. N_cycles semantics, for the full account; the module
    # (src/lifecycle.py) is retained in the codebase but is no longer invoked here or
    # reported in the manuscript.

    # ---------------------------------------------------------------
    # Manifest and log
    # ---------------------------------------------------------------
    record("fold_vs_assumed_baseline", fold_vs_base, "baseline_summary.csv", "run_all.py:1")
    with open(os.path.join(HERE, "outputs", "result_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "output_file", "produced_by"])
        w.writeheader()
        w.writerows(manifest_rows)

    total_time = time.time() - t_start
    log(f"\nTotal run_all.py wall time: {total_time:.1f} s")
    with open(os.path.join(LOGS, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
