"""Tests for observatory grading (no GPU)."""

from fpo_run_observatory.baselines import baseline_for_experiment
from fpo_run_observatory.scan import _grade


def test_baseline_lookup():
    b = baseline_for_experiment("g1_flat_flow")
    assert b is not None
    assert b.target_return == 37.0


def test_grade_cliff():
    receipt = {"cliff": {"status": "cliff_detected", "peak_reward": 35.0}}
    status, _ = _grade(receipt, baseline_for_experiment("g1_flat_flow"))
    assert status == "red"


def test_grade_healthy():
    receipt = {"cliff": {"status": "healthy", "peak_reward": 36.0}}
    status, _ = _grade(receipt, baseline_for_experiment("g1_flat_flow"))
    assert status == "green"
