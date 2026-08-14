# Toy DINOv2 prototype

## Goal

Demonstrate the main DINOv2 data flow on a tiny CPU-friendly problem:

1. make global and local crops of synthetic images;
2. train a student to match centered and sharpened EMA-teacher global outputs;
3. mask student patch tokens and match teacher patch targets;
4. apply a KoLeo-style feature-spacing regularizer;
5. verify finite losses and a non-degenerate feature spread.

## Relationship to the paper

This mirrors the conceptual combination described in *DINOv2: Learning Robust Visual Features without Supervision* (Oquab et al., arXiv:2304.07193): DINO-style global self-distillation, iBOT-style masked patch prediction, and KoLeo-style regularization. It is an educational toy, not a reproduction of LVD-142M training or the paper’s benchmark results.

## Scope and simplifications

- $16\times16$ synthetic grayscale pattern images replace the curated LVD-142M collection.
- A compact one-layer Transformer patch encoder replaces the paper’s large ViT variants.
- Random tensor crops replace the exact production augmentation and multi-crop schedule.
- Patch prediction uses a small shared head and a simple random mask.
- The KoLeo term uses nearest-neighbor distances within the current minibatch.
- Training is short and CPU-oriented; no distributed training, checkpointing, external data, or model downloads are used.

## Environment and setup

From `papers/dino_v2/`:

```bash
uv sync --locked
```

This uses the paper-local Python 3.11 environment and pinned `torch==2.7.1`.

## Run

```bash
uv run --locked python prototypes/toy_dinov2/run_toy_dinov2.py --seed 7
```

The run prints the global, patch, KoLeo, and total losses for each epoch. Use `--epochs`, `--batch-size`, and `--seed` to change the small experiment.

## Test

```bash
uv run --locked python prototypes/toy_dinov2/test_toy_dinov2.py
```

The smoke test checks output shapes, finite losses, teacher updates, and nonzero feature variance after a short run.

## Expected result

The losses should remain finite, the teacher parameters should move after an EMA update, and the feature standard deviation should be nonzero. Exact loss values are seed- and version-dependent. A toy run does not establish DINOv2’s downstream accuracy or scaling claims.

## Seed and configuration

The default seed is `7`. Images are $16\times16$, patch size is 4, the synthetic batch contains repeated stripe and blob patterns, and half of the patch tokens are masked for the patch objective.

## Limitations

The prototype demonstrates the learning signals and update order only. It does not reproduce the authors’ data curation, model scale, augmentation schedule, optimizer schedule, distributed training, or evaluation suite. Its features should be treated as teaching artifacts, not pretrained DINOv2 weights.
