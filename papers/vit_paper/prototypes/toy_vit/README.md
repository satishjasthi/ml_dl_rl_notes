# Toy Vision Transformer

## Goal

Make the ViT data path concrete on a tiny CPU-friendly task:

1. split an image into non-overlapping patches;
2. linearly embed flattened patches;
3. prepend a learnable class token;
4. add learnable positional embeddings;
5. run a Transformer encoder;
6. classify from the final class-token representation.

## Relationship to the paper

This mirrors the pure patch-to-Transformer architecture in *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., arXiv:2010.11929, Section 3.1). It is an educational toy, not a reproduction of the paper's ImageNet/JFT pretraining or scaling experiments.

## Scope and simplifications

- $8 \times 8$ grayscale synthetic images replace natural RGB images.
- $2 \times 2$ patches create 16 image tokens.
- One small Transformer encoder layer replaces the paper's deep model variants.
- A supervised stripe-classification task replaces large-scale pretraining and transfer learning.
- The implementation uses PyTorch's standard Transformer encoder; it does not reproduce every paper training detail or benchmark protocol.

## Environment and setup

From `papers/vit_paper/`:

```bash
uv sync --locked
```

This uses the pinned paper-local `torch==2.7.1` dependency. No dataset or model weights are downloaded.

## Run

```bash
uv run --locked python prototypes/toy_vit/run_toy_vit.py --seed 7
```

Optional arguments include `--epochs` and `--batch-size`.

## Test

```bash
uv run --locked python prototypes/toy_vit/test_toy_vit.py
```

The smoke test checks patch/token shapes and finite training metrics on a short run.

## Expected result

The command prints the final training loss and test accuracy. The synthetic vertical-versus-horizontal stripe task should be learnable, but exact metrics depend on the seed, PyTorch version, and epoch count. The important observation is the sequence construction and class-token readout.

## Seed and configuration

Default seed: `7`. Image size: `8`; patch size: `2`; channels: `1`; patch count: `16`; embedding width: `32`; heads: `4`; encoder depth: `1`; classes: `2`.

## Limitations

This prototype does not establish the paper's accuracy, scaling, transfer, or compute claims. It omits large-scale pretraining, resolution transfer, position interpolation, hybrid CNN stems, distributed training, and natural-image failure modes. Use it to reason about tensor flow, not as a benchmark implementation.
