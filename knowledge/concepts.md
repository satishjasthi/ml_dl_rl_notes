# Concepts

- **Contrastive self-supervised learning** — learn representations by matching related views and separating unrelated examples. Source: [SimCLR notes](../papers/simCLR/notes.md).
- **Augmentation-induced invariance** — transformations define which visual differences the representation should ignore. Source: [SimCLR notes](../papers/simCLR/notes.md).
- **NT-Xent / InfoNCE-style objective** — temperature-scaled classification of a positive view among in-batch candidates. Source: [SimCLR notes](../papers/simCLR/notes.md).
- **Projection head** — a train-time nonlinear mapping that can improve the encoder representation while being discarded downstream. Source: [SimCLR notes](../papers/simCLR/notes.md).

- **SimCLRv2 three-stage semi-supervised pipeline** — contrastive pretraining, few-label fine-tuning, and distillation with unlabeled examples. Source: [SimCLRv2 notes](../papers/simCLRv2/notes.md).
- **Soft-target knowledge distillation** — a fine-tuned teacher transfers task-specific class probabilities to a student using unlabeled examples. Source: [SimCLRv2 notes](../papers/simCLRv2/notes.md).
- **Self-supervised scaling in low-label learning** — larger contrastive models can provide stronger representations for few-label downstream tasks. Source: [SimCLRv2 notes](../papers/simCLRv2/notes.md).

- **Masked autoencoding** — reconstruct hidden image patches to learn visual representations without labels. Source: [MAE notes](../papers/masked_auto_encoders/notes.md).
- **Asymmetric visible-token encoder-decoder** — process only visible patches with a large encoder and use a lightweight decoder to reconstruct the full patch sequence. Source: [MAE notes](../papers/masked_auto_encoders/notes.md).
- **High-ratio image masking** — masking about 75% of patches makes reconstruction meaningful while reducing encoder token computation in the studied setup. Source: [MAE notes](../papers/masked_auto_encoders/notes.md).
- **Masked prediction versus contrastive learning** — MAE reconstructs missing content, whereas SimCLRv2 matches augmented views and later distills task predictions. Source: [MAE notes](../papers/masked_auto_encoders/notes.md) and [SimCLRv2 notes](../papers/simCLRv2/notes.md).

- **Self-distillation without labels** — DINO’s student learns from soft targets produced by an EMA teacher rather than human labels. Source: [DINO notes](../papers/dino-v1/notes.md).
- **Multi-crop consistency** — DINO matches global and local views of one image to encourage crop- and scale-stable features. Source: [DINO notes](../papers/dino-v1/notes.md).
- **Negative-free representation learning** — DINO avoids explicit contrastive negatives by using teacher-student cross-entropy. Source: [DINO notes](../papers/dino-v1/notes.md).
- **Emergent ViT attention structure** — DINO reports object-aligned attention behavior in self-supervised Vision Transformers. Source and qualification: [DINO notes](../papers/dino-v1/notes.md).

- **DINOv2 global-plus-patch self-distillation** — combine DINO-style global view agreement with iBOT-style masked patch prediction. Source: [DINOv2 notes](../papers/dino_v2/notes.md).
- **KoLeo feature regularization** — encourage normalized features to remain spread through nearest-neighbor distance regularization. Source: [DINOv2 notes](../papers/dino_v2/notes.md).
- **Curated visual pretraining data** — DINOv2 treats data diversity and curation as part of general-purpose feature learning. Source: [DINOv2 notes](../papers/dino_v2/notes.md).
- **DINOv1-to-DINOv2 scaling** — DINOv2 retains the EMA teacher intuition while extending the objective, data, model, and evaluation recipe. Sources: [DINOv1 notes](../papers/dino-v1/notes.md) and [DINOv2 notes](../papers/dino_v2/notes.md).
