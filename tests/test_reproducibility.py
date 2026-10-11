import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation import run_simulation


def test_deterministic_rerun_same_seed():
    out1 = run_simulation(50_000, seed=123)
    out2 = run_simulation(50_000, seed=123)
    assert np.array_equal(out1["P_TR"], out2["P_TR"])


def test_different_seed_gives_different_result():
    out1 = run_simulation(50_000, seed=123)
    out2 = run_simulation(50_000, seed=124)
    assert not np.array_equal(out1["P_TR"], out2["P_TR"])


def test_dependent_scenario_reproducible():
    out1 = run_simulation(50_000, seed=55, dependence_scenario="base")
    out2 = run_simulation(50_000, seed=55, dependence_scenario="base")
    assert np.array_equal(out1["P_TR"], out2["P_TR"])
