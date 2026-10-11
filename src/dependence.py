"""Iman-Conover rank-correlation induction for the seven input variables.

Reviewers 1 and 2 both note that the independence assumption between
SOC, C_rate, T_bat, T_amb, N_cycles, R_int0 and sigma_V is physically
debatable. We did not identify a publicly available eVTOL-specific
correlation matrix, so we do NOT present a fabricated "measured" correlation
matrix. Instead we define low/base/high illustrative dependence
scenarios over a small number of illustrative, operationally motivated
pairwise correlation signs, apply the Iman-Conover (1982) rank-correlation
procedure, and report how much the results shift relative to the
independent baseline. The correlation magnitudes are scenario
parameters, not measurements.

IMPORTANT (found during a later manuscript-consistency audit): the values in
_PAIRS below are Pearson correlations of the *latent multivariate-normal
reference* Iman-Conover builds internally (see iman_conover()'s Cholesky
step), NOT the target Spearman rank correlation of the final reordered data.
Because rank order is copied exactly from that latent reference, the
resulting Spearman correlation of the output equals the classical
Gaussian-copula transform of the latent value, rho_S = (6/pi)*arcsin(rho/2),
which is systematically smaller in magnitude (e.g. latent 0.30 -> achieved
Spearman ~0.288). The manuscript's Table 2 reports both the latent input
and the achieved Spearman correlation explicitly for this reason -- do not
describe the _PAIRS values as "target Spearman correlations" in the paper.

Assumed pairs and rationale (all illustrative, not measured):
  - T_amb vs T_bat: positive. Hot ambient conditions reduce convective/
    thermal-management cooling effectiveness between flight and charge
    initiation.
  - C_rate vs T_bat: positive. Operationally aggressive mission profiles
    (high hover/climb power) plausibly co-occur with operator/BMS
    preference for faster subsequent turnaround charging.
  - N_cycles vs SOC0: negative in the low/base/high scenarios (operators
    may adopt more conservative, lower charge ceilings for more heavily
    aged packs -- a risk-*reducing* structure, since it anti-correlates
    two of the model's largest risk drivers) and positive in the
    "adverse" scenario (aged, heavily utilized packs may instead be
    charged to higher SOC more often, e.g. if operators do not adjust
    charging behavior for pack age, or if high aircraft utilization
    drives both faster aging and less conservative charging -- a
    risk-*increasing* structure of comparable plausibility and
    magnitude). The low/base/high scenarios alone only vary correlation
    *magnitude* along one net direction; "adverse" tests the opposite
    net direction so that the reported dependence sensitivity is not
    confined to a single sign combination.
"""
from __future__ import annotations

import numpy as np

VARIABLE_ORDER = ["SOC0", "N_cycles", "T_bat", "T_amb", "C_rate", "R_int0", "sigma_V"]

_PAIRS = {
    "low": {("T_amb", "T_bat"): 0.15, ("C_rate", "T_bat"): 0.10, ("N_cycles", "SOC0"): -0.05},
    "base": {("T_amb", "T_bat"): 0.30, ("C_rate", "T_bat"): 0.20, ("N_cycles", "SOC0"): -0.10},
    "high": {("T_amb", "T_bat"): 0.50, ("C_rate", "T_bat"): 0.35, ("N_cycles", "SOC0"): -0.20},
    "adverse": {("T_amb", "T_bat"): 0.30, ("C_rate", "T_bat"): 0.20, ("N_cycles", "SOC0"): 0.30},
}


def target_correlation_matrix(scenario: str) -> np.ndarray:
    """Build a positive semi-definite target Spearman rank-correlation matrix."""
    p = len(VARIABLE_ORDER)
    idx = {v: i for i, v in enumerate(VARIABLE_ORDER)}
    m = np.eye(p)
    if scenario == "independent":
        return m
    if scenario not in _PAIRS:
        raise ValueError(f"unknown dependence scenario: {scenario}")
    for (a, b), rho in _PAIRS[scenario].items():
        i, j = idx[a], idx[b]
        m[i, j] = rho
        m[j, i] = rho
    # Ensure positive semi-definiteness: clip negative eigenvalues then re-symmetrize.
    eigval, eigvec = np.linalg.eigh(m)
    eigval_clipped = np.clip(eigval, 1e-8, None)
    m_psd = eigvec @ np.diag(eigval_clipped) @ eigvec.T
    d = np.sqrt(np.diag(m_psd))
    m_psd = m_psd / np.outer(d, d)
    np.fill_diagonal(m_psd, 1.0)
    return m_psd


def iman_conover(samples: dict, target_corr: np.ndarray, rng: np.random.Generator) -> dict:
    """Reorder samples so their rank correlation approximates `target_corr`.

    samples: dict[name] -> 1D array (already correctly drawn from its marginal).
    Returns a new dict with the same marginal values, permuted.
    Reference: Iman & Conover (1982), "A distribution-free approach to
    inducing rank correlation among input variables."
    """
    names = VARIABLE_ORDER
    n = len(samples[names[0]])
    p = len(names)

    # Step 1: generate reference normal scores with the target correlation via Cholesky.
    l_chol = np.linalg.cholesky(target_corr)
    z = rng.standard_normal(size=(n, p))
    z_corr = z @ l_chol.T

    # Step 2: rank each reference column; use these ranks to reorder each input's
    # already-sorted marginal sample, inducing the same rank correlation structure.
    out = {}
    for j, name in enumerate(names):
        ref_ranks = np.argsort(np.argsort(z_corr[:, j]))
        sorted_vals = np.sort(samples[name])
        out[name] = sorted_vals[ref_ranks]
    return out
