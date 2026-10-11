import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import risk_functions as rf


def test_f_soc_bounds_and_continuity():
    x = np.array([0.0, 29.999, 30.0, 69.999, 70.0, 95.0])
    y = rf.f_soc(x)
    assert np.all(y >= 0.5) and np.all(y <= 7.0001)
    # continuity at boundaries (piecewise pieces should nearly match at the seams)
    assert abs(rf.f_soc(np.array([30.0]))[0] - 1.0) < 1e-9
    assert abs(rf.f_soc(np.array([70.0]))[0] - 3.0) < 1e-9


def test_f_cycles_disabled_returns_one():
    n = np.array([0.0, 500.0, 3000.0])
    assert np.allclose(rf.f_cycles(n, enabled=False), 1.0)
    assert rf.f_cycles(np.array([1000.0]))[0] > 1.0


def test_t_onset_matches_calibration_point():
    # T_onset(0.5C) must equal 297.5 C exactly (the cited calibration point)
    t = rf.t_onset(np.array([0.5]), mode="linear_extrapolated")
    assert abs(t[0] - 297.5) < 1e-9
    t3 = rf.t_onset(np.array([3.0]), mode="linear_extrapolated")
    assert abs(t3[0] - 264.7) < 1e-6


def test_t_onset_flat_cap_beyond_domain():
    t_at_3 = rf.t_onset(np.array([3.0]), mode="flat_cap")[0]
    t_at_6 = rf.t_onset(np.array([6.0]), mode="flat_cap")[0]
    assert abs(t_at_3 - t_at_6) < 1e-9  # flat beyond the calibrated 3.0C point


def test_f_thermal_continuity_at_boundaries():
    m1 = rf.f_thermal(np.array([200.0 + 1e-6]))[0]
    m2 = rf.f_thermal(np.array([200.0 - 1e-6]))[0]
    assert abs(m1 - m2) < 1e-4
    m3 = rf.f_thermal(np.array([100.0 + 1e-6]))[0]
    m4 = rf.f_thermal(np.array([100.0 - 1e-6]))[0]
    assert abs(m3 - m4) < 1e-3


def test_f_crate_disabled_returns_one():
    c = np.array([1.0, 3.0, 5.0])
    assert np.allclose(rf.f_crate(c, enabled=False), 1.0)


def test_r_int_aged_disabled_ignores_cycles():
    r0 = np.array([50.0, 60.0])
    n = np.array([0.0, 3000.0])
    assert np.allclose(rf.r_int_aged(r0, n, enabled=False), r0)
    assert rf.r_int_aged(r0, n, enabled=True)[1] > r0[1]


def test_compute_ptr_positive_and_finite():
    rng = np.random.default_rng(0)
    size = 10_000
    inputs = {
        "SOC0": rng.uniform(30, 95, size),
        "N_cycles": rng.uniform(0, 5000, size),
        "T_bat": rng.uniform(35, 65, size),
        "T_amb": rng.uniform(0, 45, size),
        "C_rate": rng.uniform(3, 6, size),
        "R_int0": rng.uniform(30, 90, size),
        "sigma_V": rng.uniform(0, 100, size),
    }
    out = rf.compute_ptr(inputs)
    assert np.all(np.isfinite(out["P_TR"]))
    assert np.all(out["P_TR"] > 0)
