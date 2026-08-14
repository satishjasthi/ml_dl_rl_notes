# Methods

- **SimCLR** — simple contrastive pretraining with two augmented views per image, an encoder, an MLP projection head, NT-Xent loss, and in-batch negatives. Source: [SimCLR notes](../papers/simCLR/notes.md).

- **SimCLRv2 semi-supervised workflow** — extends [SimCLR](../papers/simCLR/notes.md) with large-model contrastive pretraining, supervised fine-tuning on few labels, and distillation using unlabeled data. Source: [SimCLRv2 notes](../papers/simCLRv2/notes.md).
- **Teacher-student distillation after self-supervised pretraining** — specialize a pretrained teacher with labels, then transfer its task predictions to a student. Source: [SimCLRv2 notes](../papers/simCLRv2/notes.md).

- **Masked autoencoder pretraining** — patchify an image, randomly hide most patches, encode only visible tokens, insert mask tokens, and reconstruct masked pixels with a lightweight decoder. Source: [MAE notes](../papers/masked_auto_encoders/notes.md).
- **Asymmetric encoder-decoder pretraining** — use a large compute-saving visible-token encoder and a small full-sequence reconstruction decoder. Source: [MAE notes](../papers/masked_auto_encoders/notes.md).

- **DINO self-distillation** — train a student to match a slowly updated EMA teacher across global and local crops using soft-target cross-entropy, without labels or explicit negatives. Source: [DINO notes](../papers/dino-v1/notes.md).
- **Teacher centering and sharpening** — stabilize DINO targets and reduce collapse risk through a running teacher-output center and temperature-scaled distributions. Source: [DINO notes](../papers/dino-v1/notes.md).

- **DINOv2 self-supervised feature learning** — scale DINO-style EMA-teacher distillation with curated LVD-142M data, iBOT-style masked patch prediction, KoLeo regularization, and large Vision Transformers. Source: [DINOv2 notes](../papers/dino_v2/notes.md).
- **Global-plus-patch distillation** — use global crop-level targets and masked patch-level targets to preserve both semantic and spatial information. Source: [DINOv2 notes](../papers/dino_v2/notes.md).

- **Transformer encoder-decoder** — sequence-transduction architecture based on self-attention, encoder-decoder attention, positional encodings, feed-forward layers, residual connections, and normalization, without recurrence or convolution. Source: [Attention Is All You Need notes](../papers/attention-is-all-you-need/notes.md).
