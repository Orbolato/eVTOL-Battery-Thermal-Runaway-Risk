"""Sensitivity analysis: Spearman correlation with bootstrap CIs, and variance-based
(Sobol) first-/total-order indices for the independent baseline model.

Reviewer 1 Concern 6 and Reviewer 2 Concern 8: Spearman rank correlation
only captures monotonic marginal association and, at N=10^7, produces
p-values that are uninformative (almost any nonzero correlation is
"significant"). We therefore (a) report bootstrap confidence intervals
on rho_s instead of emphasizing p-values, and (b) add Sobol first- and
total-order indices, which quantify each input's share of output
variance including interaction (higher-order) effects. Sobol indices
require independent inputs, so they are computed on the independent
baseline model only; for the dependent-input scenarios we instead
report a structural-ablation comparison (Sec. 7.3) and permutation
importance, which remain interpretable under dependence.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from .simulation import sample_inputs
from .risk_functions import compute_ptr, AblationConfig, FULL_MODEL

INPUT_NAMES = ["SOC0", "N_cycles", "T_bat", "T_amb", "C_rate", "R_int0", "sigma_V"]


def spearman_bootstrap(inputs: dict, ptr: np.ndarray, n_boot: int = 500,
                        seed: int = 20260716) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(ptr)
    rows = []
    for name in INPUT_NAMES:
        x = inputs[name]
        rho, pval = stats.spearmanr(x, ptr)
        boot_rhos = np.empty(n_boot)
        # Sub-sample bootstrap (size 20,000) for tractable runtime at N up to 1e7.
        sub_n = min(20_000, n)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=sub_n)
            boot_rhos[b] = stats.spearmanr(x[idx], ptr[idx])[0]
        lo, hi = np.percentile(boot_rhos, [2.5, 97.5])
        rows.append({
            "variable": name,
            "rho_s": float(rho),
            "p_value": float(pval),
            "ci95_lo": float(lo),
            "ci95_hi": float(hi),
        })
    rows.sort(key=lambda r: -abs(r["rho_s"]))
    return rows


def _sample_matrix(size: int, rng: np.random.Generator) -> np.ndarray:
    """Return an (size, 7) array of independent draws, one column per input, in
    INPUT_NAMES order, as raw uniforms mapped through each marginal via sample_inputs."""
    d = sample_inputs(size, rng, dependence_scenario="independent")
    return np.column_stack([d[name] for name in INPUT_NAMES])


def _ptr_from_matrix(mat: np.ndarray, ablation: AblationConfig) -> np.ndarray:
    d = {name: mat[:, i] for i, name in enumerate(INPUT_NAMES)}
    return compute_ptr(d, ablation=ablation)["P_TR"]


def sobol_indices(size: int = 200_000, seed: int = 20260716,
                   ablation: AblationConfig = FULL_MODEL) -> list[dict]:
    """Saltelli (2002) first-order and Jansen (1999) total-order Sobol index
    estimators (matrices A, B, and AB^i = A with column i replaced by B's column i;
    see Saltelli et al. 2008, "Global Sensitivity Analysis: The Primer", eqs.
    4.14-4.18). Requires independent inputs (used only for the independent baseline
    model, per the module docstring).

    S_i  = E[ y_B * (y_{AB^i} - y_A) ] / Var(Y)                  (first order)
    ST_i = E[ (y_A - y_{AB^i})^2 ] / (2 * Var(Y))                (total order)
    """
    rng = np.random.default_rng(seed)
    a_mat = _sample_matrix(size, rng)
    b_mat = _sample_matrix(size, rng)

    ya = _ptr_from_matrix(a_mat, ablation)
    yb = _ptr_from_matrix(b_mat, ablation)
    var_y = np.var(np.concatenate([ya, yb]), ddof=1)

    rows = []
    for i, name in enumerate(INPUT_NAMES):
        ab_i = a_mat.copy()
        ab_i[:, i] = b_mat[:, i]
        y_ab_i = _ptr_from_matrix(ab_i, ablation)

        s1 = np.mean(yb * (y_ab_i - ya)) / var_y
        st = np.mean((ya - y_ab_i) ** 2) / (2.0 * var_y)
        rows.append({"variable": name, "S1": float(s1), "ST": float(st)})

    rows.sort(key=lambda r: -r["ST"])
    return rows
