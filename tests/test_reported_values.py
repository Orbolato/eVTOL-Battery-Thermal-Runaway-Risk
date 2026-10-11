"""Sanity checks on the values that run_all.py writes to outputs/tables and that are
quoted verbatim in the revised manuscript and response to reviewers. These tests do
not hardcode manuscript numbers (that would be circular, since the manuscript numbers
are generated from these same tables); instead they check internal consistency and
plausibility bounds so a future code change cannot silently produce nonsensical
numbers that still "look like" a plausible P_TR.
"""
import os
import csv
import pytest

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "tables")


def _read_csv(name):
    path = os.path.join(OUT_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not yet generated; run code/run_all.py first")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_baseline_summary_plausible():
    rows = _read_csv("baseline_summary.csv")
    mean_ptr = float(rows[0]["mean"])
    assert 1e-9 < mean_ptr < 1e-5


def test_ablation_matrix_orders_sensibly():
    rows = _read_csv("ablation_matrix.csv")
    by_label = {r["label"]: float(r["mean_P_TR"]) for r in rows}
    # removing a direct pathway (holding it at neutral 1.0) must not increase risk
    assert by_label["no_direct_cycle_factor"] <= by_label["full_model"]
    assert by_label["no_direct_crate_factor"] <= by_label["full_model"]


def test_mitigation_strategies_reduce_risk_relative_to_baseline():
    rows = _read_csv("mitigation_strategies.csv")
    by_name = {r["strategy"]: r for r in rows}
    baseline = float(by_name["baseline"]["mean_P_TR"])
    for name in ("soc_85", "crate_5", "cycles_1200", "combined_moderate", "combined_aggressive"):
        assert float(by_name[name]["mean_P_TR"]) <= baseline


def test_convergence_relative_sem_shrinks_with_n():
    rows = _read_csv("convergence.csv")
    rows_sorted = sorted(rows, key=lambda r: int(r["N"]))
    rel_sems = [float(r["relative_sem_pct"]) for r in rows_sorted]
    assert rel_sems[-1] < rel_sems[0]
