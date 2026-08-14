# Toy masked autoencoder

## Goal

Demonstrate the central MAE data flow on a tiny CPU-friendly problem:

1. split images into patches;
2. hide 75% of patches;
3. encode only visible patches;
4. insert mask tokens and decode the full sequence;
5. compute reconstruction loss on masked patches;
6. evaluate the pretrained encoder with a small linear classifier.

## Relationship to the paper

This mirrors the core asymmetric encoder-decoder design in *Masked Autoencoders Are Scalable Vision Learners* (He et al., arXiv:2111.06377, Sections 1 and 3). It is an educational toy, not a reproduction of the paper's ImageNet-1K experiments.

## Scope and simplifications

- $8\times8$ grayscale synthetic stripe images replace natural images.
- $2\times2$ patches create 16 tokens rather than the paper's ImageNet-scale patch sequence.
- The encoder and decoder are one-layer, small Transformer models.
- Training is short and CPU-oriented.
- The classifier uses a frozen mean-pooled encoder representation; the paper also studies fine-tuning.
- The code uses normalized patch targets for the reconstruction loss, but does not reproduce every paper training detail.

## Environment and setup

From `papers/masked_auto_encoders/`:

```bash
uv sync --locked
```

This creates or updates the paper-local Python 3.11 environment using the pinned `torch==2.7.1` dependency. No external dataset or model weights are downloaded.

## Run

```bash
uv run --locked python prototypes/toy_mae/run_toy_mae.py --seed 7
```

Optional arguments include `--pretrain-epochs` and `--classifier-epochs`.

## Test

```bash
uv run --locked python prototypes/toy_mae/test_toy_mae.py
```

The test uses a smaller run and checks finite reconstruction loss, valid masking, and a valid classifier accuracy.

## Expected result

The command prints a decreasing or finite reconstruction loss and a classifier accuracy between 0 and 1. Exact values depend on the seed and PyTorch version. The important observation is the computation path: the encoder receives only visible tokens, while the decoder reconstructs the full sequence.

## Seed and configuration

The default seed is `7`. The default image size is 8, patch size is 2, masking ratio is 0.75, and the synthetic dataset contains horizontal and vertical stripe classes.

## Limitations

This prototype does not establish the paper's accuracy, scaling, or compute claims. It is intended to make the masking, visible-token encoding, mask-token insertion, and masked reconstruction loss concrete.
