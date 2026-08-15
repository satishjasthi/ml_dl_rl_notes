"""Smoke tests for the toy ViT."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_toy_vit import ToyViT, make_dataset, train_demo  # noqa: E402


def test_patch_and_token_shapes() -> None:
    model = ToyViT()
    images, _ = make_dataset(4)
    patches = model.patchify(images)
    logits, tokens = model(images)
    assert patches.shape == (4, 16, 4)
    assert tokens.shape == (4, 17, 32)
    assert logits.shape == (4, 2)


def test_demo_returns_finite_metrics() -> None:
    metrics = train_demo(seed=7, epochs=2, batch_size=32)
    assert set(metrics) == {"loss", "accuracy"}
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


if __name__ == "__main__":
    test_patch_and_token_shapes()
    test_demo_returns_finite_metrics()
    print("toy ViT smoke tests passed")
