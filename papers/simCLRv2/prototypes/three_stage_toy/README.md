# SimCLRv2 three-stage toy workflow

## Goal

Demonstrate the workflow proposed by SimCLRv2 on a tiny synthetic classification problem:

1. contrastive pretraining with unlabeled examples;
2. fine-tuning a teacher with a small labeled subset;
3. distilling the teacher into a smaller student using unlabeled examples.

## Relationship to the paper

This prototype mirrors the paper's high-level pipeline from *Big Self-Supervised Models are Strong Semi-Supervised Learners* (Chen et al., arXiv:2006.10029). It is not an ImageNet reproduction: it uses 2-D synthetic points, small MLPs, short training, and no large model or external dataset.

## Simplifications

- Synthetic Gaussian clusters replace images.
- The encoder and projector are small MLPs rather than large ResNet/Selective-Kernel models.
- The contrastive objective uses in-batch negatives and Gaussian perturbations as toy views.
- The student uses teacher soft targets on all unlabeled points; this is an educational distillation choice, not a claim that it exactly reproduces every paper detail.
- Accuracy is illustrative and should not be compared with the paper's ImageNet results.

## Environment and setup

From `papers/simCLRv2/`:

```bash
uv sync --locked
```

The project pins Python 3.11 and `torch==2.7.1` in the paper-level `pyproject.toml` and `uv.lock`.

## Run

```bash
uv run --locked python prototypes/three_stage_toy/run_simclrv2_toy.py --seed 7
```

The script prints the contrastive loss, the few-label teacher accuracy, and the distilled student accuracy.

## Test

```bash
uv run --locked python prototypes/three_stage_toy/test_simclrv2_toy.py
```

The smoke test checks deterministic finite outputs and verifies that every stage returns a valid accuracy.

## Expected result

A short CPU run should print three stages and accuracies in the range $[0,1]$. Exact values depend on the seed and PyTorch version. The important observation is the data flow: unlabeled views train the representation, a few labels specialize a teacher, and unlabeled examples supervise the student through teacher predictions.

## Configuration and seed

The default seed is `7`. Use `--seed` to repeat the toy experiment with another deterministic seed. The default dataset has three synthetic classes, 240 total points, and four labeled points per class.

## Limitations

This prototype validates mechanics and intuition only. It does not reproduce ImageNet data, model scale, training duration, augmentation policy, optimizer schedule, reported accuracy, or the paper's complete ablation matrix.
