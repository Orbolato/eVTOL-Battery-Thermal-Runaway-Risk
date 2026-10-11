"""Semi-empirical risk-amplification functions and structural ablation switches.

Every function is explicitly semi-empirical (see ../../Review_Response/
source_audit.csv for the evidence grade of each). None is claimed to be
calibrated or validated against eVTOL-specific data.

Structural double-counting (Reviewer 1, Concern 1; instructions Sec.
5.3): N_cycles affects thermal-runaway probability along two pathways
(direct f_cycles, and indirectly through R_int -> f_deg), and C_rate
affects it along two pathways (direct f_Crate, and indirectly through
T_onset -> f_thermal). Both pathways can be switched on/off
independently through the `AblationConfig` flags so that the resulting
double-counting can be quantified rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AblationConfig:
    """Structural-model switches used to quantify double counting (Sec. 7.3)."""
    direct_cycle_factor: bool = True   # f_cycles(N) applied directly
    rint_cycle_link: bool = True       # N_cycles increases R_int (aging pathway)
    direct_crate_factor: bool = True   # f_Crate(C) applied directly
    crate_thermal_link: bool = True    # C_rate lowers T_onset (thermal pathway)
    thermal_extrapolation: str = "flat_cap"  # "flat_cap" or "linear_extrapolated"

    label: str = "full_model"


FULL_MODEL = AblationConfig(label="full_model")
NO_DIRECT_CYCLE = AblationConfig(direct_cycle_factor=False, label="no_direct_cycle_factor")
NO_RINT_CYCLE_LINK = AblationConfig(rint_cycle_link=False, label="no_rint_cycle_link")
NO_DIRECT_CRATE = AblationConfig(direct_crate_factor=False, label="no_direct_crate_factor")
NO_CRATE_THERMAL_LINK = AblationConfig(crate_thermal_link=False, label="no_crate_thermal_link")
LINEAR_EXTRAPOLATED = AblationConfig(thermal_extrapolation="linear_extrapolated", label="linear_extrapolated_thermal")

ABLATION_SCENARIOS = [
    FULL_MODEL,
    NO_DIRECT_CYCLE,
    NO_RINT_CYCLE_LINK,
    NO_DIRECT_CRATE,
    NO_CRATE_THERMAL_LINK,
    LINEAR_EXTRAPOLATED,
]

P_BASE_ASSUMED = 8.4e-9  # per charge; treated as an ASSUMED reference value, not a
                          # verified per-charge automotive baseline (R1.3; Sec. 5.4).


def f_soc(soc: np.ndarray) -> np.ndarray:
    """Piecewise-linear SOC risk factor, f_SOC in [0.5, 7.0]. Semi-empirical (FAA
    qualitative flame observations at 3 SOC points; Sec. source_audit.csv row 'f_SOC').
    """
    soc = np.asarray(soc, dtype=float)
    out = np.empty_like(soc)
    low = soc < 30.0
    mid = (soc >= 30.0) & (soc < 70.0)
    high = soc >= 70.0
    out[low] = 0.5
    out[mid] = 1.0 + 2.0 * ((soc[mid] - 30.0) / 40.0)
    out[high] = 3.0 + 4.0 * ((soc[high] - 70.0) / 30.0)
    return out


def f_cycles(n_cycles: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Direct exponential cycle-aging risk factor. Disabled -> returns 1 (no effect),
    used to isolate the resistance-mediated aging pathway (ablation NO_DIRECT_CYCLE).
    """
    n_cycles = np.asarray(n_cycles, dtype=float)
    if not enabled:
        return np.ones_like(n_cycles)
    return np.exp(0.0005 * n_cycles)


def t_onset(c_rate: np.ndarray, mode: str = "flat_cap", enabled: bool = True,
            c_rate_for_thermal: np.ndarray | None = None) -> np.ndarray:
    """Thermal-runaway onset temperature (deg C) as a function of C-rate.

    Calibration source (Meng et al.): T_onset = 297.5 C at 0.5C and
    264.7 C at 3.0C -- a measured domain of [0.5, 3.0] C. The entire
    operational C_rate support used in this study, [3, 6] C, lies at or
    beyond the *upper edge* of that calibrated domain. The original
    manuscript's T_onset=297.5-32.8*(C/3.0) neither passes through the
    stated calibration point nor restricts its domain.

    Corrected relationship, valid on [0.5, 3.0] C only:
        T_onset(C) = 297.5 - 13.12*(C - 0.5)   for 0.5 <= C <= 3.0

    Because every sampled C_rate in this study exceeds 3.0 C, every
    T_onset value used here is necessarily an extrapolation. Two
    extrapolation scenarios are provided and compared as a structural-
    uncertainty bound (Sec. 7.3):
      - "flat_cap": T_onset held at its 3.0C calibrated value (264.7 C)
        for all C > 3.0C (conservative: does not extrapolate the trend).
      - "linear_extrapolated": the linear trend is extended beyond 3.0C
        (aggressive: assumes the measured slope holds well outside its
        calibrated range).
    """
    c = np.asarray(c_rate, dtype=float)
    c_ref = c if c_rate_for_thermal is None else np.asarray(c_rate_for_thermal, dtype=float)
    if not enabled:
        c_ref = np.full_like(c, 0.5)  # C-rate has no effect on thermal onset (ablation)

    calibrated = 297.5 - 13.12 * (np.clip(c_ref, 0.5, 3.0) - 0.5)
    if mode == "flat_cap":
        beyond = 297.5 - 13.12 * (3.0 - 0.5)  # = 264.7, constant beyond 3.0C
        return np.where(c_ref <= 3.0, calibrated, beyond)
    elif mode == "linear_extrapolated":
        return 297.5 - 13.12 * (c_ref - 0.5)
    else:
        raise ValueError(f"unknown thermal onset mode: {mode}")


def ambient_penalty(t_amb: np.ndarray) -> np.ndarray:
    t_amb = np.asarray(t_amb, dtype=float)
    return np.where(t_amb > 35.0, 0.5 * (t_amb - 35.0), 0.0)


def thermal_margin(t_onset_val: np.ndarray, t_bat: np.ndarray, t_amb: np.ndarray) -> np.ndarray:
    return t_onset_val - np.asarray(t_bat, dtype=float) - ambient_penalty(t_amb)


def f_thermal(margin: np.ndarray) -> np.ndarray:
    """Piecewise thermal-margin risk factor. Semi-empirical envelope, not a validated
    kinetic model (Sec. source_audit.csv row 'f_thermal')."""
    margin = np.asarray(margin, dtype=float)
    out = np.empty_like(margin)
    high = margin > 200.0
    mid = (margin > 100.0) & (margin <= 200.0)
    low = margin <= 100.0
    out[high] = 1.0
    out[mid] = 1.0 + 0.5 * ((200.0 - margin[mid]) / 100.0)
    out[low] = 1.5 * np.exp(0.02 * (100.0 - margin[low]))
    return out


def f_crate(c_rate: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Direct piecewise-linear C-rate risk factor. Disabled -> returns 1, used to
    isolate the thermal-onset-mediated C-rate pathway (ablation NO_DIRECT_CRATE)."""
    c = np.asarray(c_rate, dtype=float)
    if not enabled:
        return np.ones_like(c)
    out = np.empty_like(c)
    low = c <= 1.5
    mid = (c > 1.5) & (c <= 3.0)
    high = c > 3.0
    out[low] = 1.0
    out[mid] = 1.0 + 0.2 * (c[mid] - 1.5)
    out[high] = 1.3 + 0.4 * (c[high] - 3.0)
    return out


def r_int_aged(r_int0: np.ndarray, n_cycles: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Cycle-dependent internal-resistance aging. Disabled -> R_int stays at its
    fresh-cell (lognormal) value regardless of N_cycles, used to isolate the direct
    f_cycles pathway (ablation NO_RINT_CYCLE_LINK)."""
    r_int0 = np.asarray(r_int0, dtype=float)
    if not enabled:
        return r_int0
    n_cycles = np.asarray(n_cycles, dtype=float)
    return r_int0 * (1.0 + 0.5 * (n_cycles / 1000.0))


def f_uniformity(sigma_v: np.ndarray) -> np.ndarray:
    sigma_v = np.asarray(sigma_v, dtype=float)
    return np.where(sigma_v > 50.0, 1.0 + 0.01 * (sigma_v - 50.0), 1.0)


def f_deg(r_int: np.ndarray, sigma_v: np.ndarray, r_int_ref: float = 50.0) -> np.ndarray:
    r_int = np.asarray(r_int, dtype=float)
    return np.sqrt(r_int / r_int_ref) * f_uniformity(sigma_v)


def compute_ptr(inputs: dict, ablation: AblationConfig = FULL_MODEL,
                 p_base: float = P_BASE_ASSUMED) -> dict:
    """Compute P_TR and every intermediate quantity for one sample batch.

    `inputs` must contain arrays: SOC0, N_cycles, T_bat, T_amb, C_rate, R_int0, sigma_V.
    Returns a dict with P_TR and all intermediate factors (for sensitivity analysis
    and diagnostics).
    """
    soc = inputs["SOC0"]
    n = inputs["N_cycles"]
    tbat = inputs["T_bat"]
    tamb = inputs["T_amb"]
    crate = inputs["C_rate"]
    r0 = inputs["R_int0"]
    sigv = inputs["sigma_V"]

    fsoc = f_soc(soc)
    fcyc = f_cycles(n, enabled=ablation.direct_cycle_factor)

    t_on = t_onset(crate, mode=ablation.thermal_extrapolation, enabled=ablation.crate_thermal_link)
    margin = thermal_margin(t_on, tbat, tamb)
    fth = f_thermal(margin)

    fcr = f_crate(crate, enabled=ablation.direct_crate_factor)

    r_aged = r_int_aged(r0, n, enabled=ablation.rint_cycle_link)
    fdg = f_deg(r_aged, sigv)

    ptr = p_base * fsoc * fcyc * fth * fcr * fdg

    return {
        "P_TR": ptr,
        "f_SOC": fsoc,
        "f_cycles": fcyc,
        "T_onset": t_on,
        "thermal_margin": margin,
        "f_thermal": fth,
        "f_Crate": fcr,
        "R_int": r_aged,
        "f_deg": fdg,
    }
