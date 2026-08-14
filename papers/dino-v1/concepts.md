# DINO Concepts

- **Self-distillation without labels** — a student matches soft targets from a teacher derived from the student itself, with no human labels. Source: [DINO notes](notes.md).
- **EMA teacher** — teacher parameters are an exponential moving average of student parameters, producing a slowly changing target. Source: [DINO notes](notes.md).
- **Multi-crop consistency** — the student matches teacher outputs across global and local views of the same image. Source: [DINO notes](notes.md).
- **Teacher centering and sharpening** — output normalization and temperature choices that help prevent representational collapse in the studied recipe. Source: [DINO notes](notes.md).
- **Negative-free self-supervised learning** — DINO uses teacher-student cross-entropy rather than explicit in-batch negative examples. Source: [DINO notes](notes.md).
- **Emergent attention structure** — self-supervised ViT attention maps can align with objects or object parts in the paper’s experiments; this is empirical and not a universal interpretability guarantee. Source: [DINO notes](notes.md).
- **DINO versus masked prediction** — DINO learns view agreement, whereas MAE reconstructs missing image patches. Sources: [DINO notes](notes.md) and [MAE notes](../masked_auto_encoders/notes.md).
