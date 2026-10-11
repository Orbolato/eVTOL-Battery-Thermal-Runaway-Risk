import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import distributions as dist

N = 2_000_000
TOL_PCT = 1.0  # percent tolerance for Monte Carlo moment checks


def rel_err_pct(a, b):
    return 100.0 * abs(a - b) / abs(b)


def test_soc0_mean_and_support():
    rng = np.random.default_rng(1)
    x = dist.sample_soc0(N, rng)
    assert x.min() >= 30.0 and x.max() <= 95.0
    assert rel_err_pct(x.mean(), 70.625) < TOL_PCT


def test_sigma_v_mean_and_support():
    rng = np.random.default_rng(2)
    x = dist.sample_sigma_v(N, rng)
    assert x.min() >= 0.0 and x.max() <= 100.0
    assert rel_err_pct(x.mean(), 34.35) < TOL_PCT


def test_crate_mean_and_support():
    rng = np.random.default_rng(3)
    x = dist.sample_crate(N, rng)
    assert x.min() >= 3.0 and x.max() <= 6.0
    assert rel_err_pct(x.mean(), 4.81) < TOL_PCT


def test_ncycles_mean_cap_and_negligible_truncation():
    rng = np.random.default_rng(4)
    x = dist.sample_ncycles(N, rng)
    assert x.min() >= 0.0 and x.max() <= 5000.0
    # untruncated P(N>5000) is ~1e-15; confirm no meaningful mass piles at the cap
    assert (x > 4990).mean() < 1e-3
    # untruncated Weibull(k=2.5, lambda=1200) analytic mean is
    # lambda*Gamma(1+1/k) = 1200*Gamma(1.4) ~= 1064.5 cycles; the negligible cap
    # leaves this numerically unchanged, closing the previously unverified "--"
    # entry in the manuscript's Table I (resubmission audit item 5).
    assert rel_err_pct(x.mean(), 1064.48) < TOL_PCT


def test_tbat_support():
    rng = np.random.default_rng(5)
    x = dist.sample_tbat(N, rng)
    assert x.min() >= 35.0 and x.max() <= 65.0
    assert rel_err_pct(x.mean(), 50.0) < TOL_PCT


def test_tamb_support():
    rng = np.random.default_rng(6)
    x = dist.sample_tamb(N, rng)
    assert x.min() >= 0.0 and x.max() <= 45.0
    # Asymmetric truncation ([0,45] around mu=25, sigma=10 is NOT symmetric: -2.5
    # sigma on the left, +2.0 sigma on the right) shifts the true conditional mean
    # to ~24.62 degC, not the input mu=25 -- analogous to the other truncated-
    # distribution reporting errors documented in baseline_audit.md.
    assert rel_err_pct(x.mean(), 24.6245) < TOL_PCT


def test_rint0_lognormal_mean():
    rng = np.random.default_rng(7)
    x = dist.sample_rint0(N, rng)
    expected_mean = 50.0 * np.exp(0.3 ** 2 / 2)
    assert rel_err_pct(x.mean(), expected_mean) < TOL_PCT
