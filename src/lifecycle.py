"""Sequential (age-evolving) life-cycle simulation vs. constant-rate Bernoulli/Poisson
approximations (instructions Sec. 5.6).

The original manuscript applied one population-average per-charge
probability P_TR-bar uniformly across a battery's life to obtain annual
and fleet-level risk, even though N_cycles (and hence R_int and
f_cycles) increase with every charge. This module instead simulates
each aircraft's full charge history explicitly (N_cycles incrementing
by exactly 1 per charge, R_int aging deterministically with it,
SOC0/T_bat/T_amb/C_rate/sigma_V resampled i.i.d. per charge as
operational variability) and compares three aggregation schemes for
P(>=1 thermal-runaway event) over a fixed number of charges per
aircraft:

  1. "sequential_exact": product of (1 - P_TR(t)) over the aircraft's
     actual age-evolving per-charge probabilities (Bernoulli, exact,
     age-dependent).
  2. "constant_rate_bernoulli": 1 - (1 - P_TR-bar)^N_charges, using the
     population mean per-charge probability held constant (the original
     manuscript's implicit approach).
  3. "constant_rate_poisson": 1 - exp(-N_charges * P_TR-bar), the
     Poisson approximation to (2).

Comparing (1) to (2)/(3) isolates the error introduced by ignoring
within-life aging; comparing (2) to (3) isolates the (expected to be
negligible, at these probability magnitudes) Bernoulli-vs-Poisson
approximation error.

Pack replacement: N_cycles is described in the manuscript as following
a Weibull distribution "characteristic of wear-out failure modes" --
i.e., Weibull(2.5, 1200) is the pack's cycle-life-to-replacement
distribution, not an unbounded counter. Earlier development versions of
this module let N_cycles grow without bound across an aircraft's full
multi-year charge history, which drove the exponential f_cycles(N) and
the linear R_int(N) terms to implausible values (confirmed by a sanity
check: an early run produced expected fires ~= fleet size, i.e. every
aircraft catching fire, which is inconsistent with any of this study's
per-charge probabilities and reveals a modeling error, not a
finding). Each simulated aircraft/pack here instead draws its own
replacement-cycle threshold from Weibull(2.5, 1200) (capped at 5000,
consistent with distributions.sample_ncycles) and is reset (fresh
R_int0 draw, cycle counter back to 1) whenever its accumulated cycle
count would exceed that threshold, representing a scheduled
replacement-before-end-of-life policy.
"""
from __future__ import annotations

import numpy as np

from . import distributions as dist
from .risk_functions import compute_ptr, FULL_MODEL, P_BASE_ASSUMED

REPLACEMENT_CAP = 5000.0


def simulate_fleet_sequential(n_aircraft: int, n_charges: int, seed: int) -> np.ndarray:
    """Return an (n_aircraft,) array of per-aircraft P(>=1 fire) over n_charges
    sequential, age-evolving charges, with pack replacement at each aircraft's own
    Weibull-drawn cycle-life threshold (see module docstring)."""
    rng = np.random.default_rng(seed)
    r_int0 = dist.sample_rint0(n_aircraft, rng)
    replace_at = dist.sample_ncycles(n_aircraft, rng, cap=REPLACEMENT_CAP)
    replace_at = np.maximum(replace_at, 1.0)  # avoid a degenerate zero-cycle threshold
    n_cycles = np.ones(n_aircraft)

    log_survival = np.zeros(n_aircraft)
    for _ in range(1, n_charges + 1):
        due = n_cycles > replace_at
        if due.any():
            r_int0[due] = dist.sample_rint0(int(due.sum()), rng)
            replace_at[due] = np.maximum(
                dist.sample_ncycles(int(due.sum()), rng, cap=REPLACEMENT_CAP), 1.0)
            n_cycles[due] = 1.0

        inputs_t = {
            "SOC0": dist.sample_soc0(n_aircraft, rng),
            "N_cycles": n_cycles.copy(),
            "T_bat": dist.sample_tbat(n_aircraft, rng),
            "T_amb": dist.sample_tamb(n_aircraft, rng),
            "C_rate": dist.sample_crate(n_aircraft, rng),
            "R_int0": r_int0,
            "sigma_V": dist.sample_sigma_v(n_aircraft, rng),
        }
        ptr_t = compute_ptr(inputs_t, ablation=FULL_MODEL)["P_TR"]
        log_survival += np.log1p(-ptr_t)
        n_cycles += 1.0

    return 1.0 - np.exp(log_survival)


def compare_aggregation_schemes(n_aircraft: int, n_charges: int, p_tr_bar: float,
                                 seed: int = 20260716) -> dict:
    p_seq = simulate_fleet_sequential(n_aircraft, n_charges, seed)
    mean_p_seq = float(np.mean(p_seq))

    p_bernoulli_const = 1.0 - (1.0 - p_tr_bar) ** n_charges
    p_poisson_const = 1.0 - np.exp(-n_charges * p_tr_bar)

    return {
        "n_aircraft": n_aircraft,
        "n_charges": n_charges,
        "p_tr_bar_used": p_tr_bar,
        "sequential_exact_mean": mean_p_seq,
        "constant_rate_bernoulli": p_bernoulli_const,
        "constant_rate_poisson": p_poisson_const,
        "bernoulli_vs_poisson_abs_error": abs(p_bernoulli_const - p_poisson_const),
        "constant_rate_vs_sequential_abs_error": abs(p_bernoulli_const - mean_p_seq),
        "constant_rate_vs_sequential_rel_error_pct":
            100.0 * (p_bernoulli_const - mean_p_seq) / mean_p_seq if mean_p_seq > 0 else float("nan"),
    }
