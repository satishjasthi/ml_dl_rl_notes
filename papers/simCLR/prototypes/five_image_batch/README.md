# Five-image SimCLR batch walkthrough

## Goal

Demonstrate one complete SimCLR forward pass using exactly five public internet images:

1. Download/cache five images.
2. Create two independently augmented views per image.
3. Build a batch of ten views.
4. Save `view_1.png` and `view_2.png` under `views/<image_name>/` for every source image.
5. Encode the views with a randomly initialized ResNet-18.
6. Apply a two-layer MLP projection head.
7. Compute cosine similarities and the bidirectional NT-Xent loss.
8. Print shapes, positive-pair labels, probabilities, and the loss for one anchor.

## Relationship to the paper

This is a small educational implementation of the method described in *A Simple Framework for Contrastive Learning of Visual Representations* (Chen et al., arXiv:2002.05709), especially the data augmentation, encoder/projector, and NT-Xent components.

## Environment

The paper-local environment is defined by:

- `../pyproject.toml`
- `../uv.lock`
- Python 3.11
- `torch==2.7.1`
- `torchvision==0.22.1`
- `Pillow==11.3.0`

The environment is created at `../.venv/` and should be used through `uv run --locked`.

## Run

From `papers/simCLR/`:

```bash
uv run --locked python prototypes/five_image_batch/run_simclr_batch.py
```

The first run downloads five images into `prototypes/five_image_batch/data/`; that directory is ignored by Git. The generated views are saved as `prototypes/five_image_batch/views/<image_name>/view_1.png` and `view_2.png`. The views directory is also ignored by Git. Later runs reuse the cached source images and overwrite the saved view files. Use `--seed` to choose a different reproducible subset from the URL pool:

```bash
uv run --locked python prototypes/five_image_batch/run_simclr_batch.py --seed 11
```

## Expected result

The script prints:

- five downloaded or cached images;
- two tensors with shape `(5, 3, 128, 128)`;
- ten combined views with shape `(10, 3, 128, 128)`;
- encoder outputs `h` with shape `(10, 512)`;
- projection outputs `z` with shape `(10, 128)`;
- positive labels such as `5` for anchor `0`;
- a scalar NT-Xent loss.

Exact loss values vary with the selected images and random augmentations.

## Simplifications and limitations

- The ResNet-18 encoder is randomly initialized, not ImageNet-pretrained, because the purpose is to demonstrate the mechanics of SimCLR pretraining.
- This performs one forward pass only; it does not train the encoder.
- The batch has only five original images, whereas the paper benefits from much larger batches and long training.
- The public image URLs are a convenient teaching source, not a curated benchmark dataset.
- The implementation uses a simple two-layer MLP projector and does not reproduce every paper training detail such as LARS optimization or ImageNet-scale settings.
