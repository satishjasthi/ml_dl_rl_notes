"""Smoke test for the SimCLRv2 toy workflow."""

from __future__ import annotations

import math

from run_simclrv2_toy import run


def test_three_stage_workflow_returns_finite_metrics() -> None:
    metrics = run(seed=7)
    assert 0.0 <= metrics["teacher_accuracy"] <= 1.0
    assert 0.0 <= metrics["student_accuracy"] <= 1.0
    assert math.isfinite(metrics["contrastive_loss"])


if __name__ == "__main__":
    test_three_stage_workflow_returns_finite_metrics()
    print("smoke test passed")
