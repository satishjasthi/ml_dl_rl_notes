"""Smoke tests for the toy MAE."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_toy_mae import MaskedAutoencoder, make_dataset, train_demo  # noqa: E402


def test_mask_shape_and_ratio() -> None:
    model = MaskedAutoencoder()
    images, _ = make_dataset(4)
    _, _, mask = model(images)
    assert mask.shape == (4, 16)
    assert mask.sum(dim=1).tolist() == [12, 12, 12, 12]


def test_demo_returns_finite_metrics() -> None:
    metrics = train_demo(seed=7, pretrain_epochs=2, classifier_epochs=3)
    assert set(metrics) == {"reconstruction_loss", "linear_accuracy"}
    assert math.isfinite(metrics["reconstruction_loss"])
    assert 0.0 <= metrics["linear_accuracy"] <= 1.0


if __name__ == "__main__":
    test_mask_shape_and_ratio()
    test_demo_returns_finite_metrics()
    print("toy MAE smoke tests passed")
