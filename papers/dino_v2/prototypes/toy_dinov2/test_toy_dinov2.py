"""Smoke tests for the toy DINOv2 prototype."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_toy_dinov2 import ToyTrainer, make_synthetic_images, make_views, run  # noqa: E402


def test_shapes_and_one_step() -> None:
    images = make_synthetic_images(4)
    views = make_views(images, seed=7)
    trainer = ToyTrainer()
    before = [parameter.detach().clone() for parameter in trainer.teacher.parameters()]
    metrics = trainer.step(views, seed=8)
    assert all(math.isfinite(value) for value in metrics.values())
    assert trainer.student(images)[0].shape == (4, 32)
    assert trainer.student(images)[1].shape == (4, 16, 32)
    assert any(not torch.equal(old, new) for old, new in zip(before, trainer.teacher.parameters()))


def test_short_run() -> None:
    metrics = run(epochs=1, batch_size=12, seed=7)
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["feature_std"] > 1e-5
    assert metrics["teacher_student_distance"] > 0.0


if __name__ == "__main__":
    test_shapes_and_one_step()
    test_short_run()
    print("toy DINOv2 smoke tests passed")
