# DINOv2 Concepts

- **Self-distillation without labels** — a student matches soft targets from a slowly updated teacher across views. Source: [DINOv2 notes](notes.md).
- **EMA teacher** — teacher parameters are an exponential moving average of student parameters, producing a slowly changing target. Source: [DINOv2 notes](notes.md); compare [DINOv1 notes](../dino-v1/notes.md).
- **Global/local crop consistency** — global teacher views supervise student global and local views. Source: [DINOv2 notes](notes.md).
- **iBOT-style masked patch prediction** — student patch tokens predict teacher targets for masked regions, complementing global image-level distillation. Source: [DINOv2 notes](notes.md).
- **KoLeo regularization** — nearest-neighbor feature-spacing regularization intended to keep normalized features spread out. Source: [DINOv2 notes](notes.md).
- **Teacher centering and sharpening** — output centering and temperature choices that help stabilize self-distillation and reduce collapse risk. Source: [DINOv2 notes](notes.md); compare [DINOv1 notes](../dino-v1/notes.md).
- **Curated self-supervised data** — LVD-142M illustrates that data diversity and curation are part of DINOv2’s scaling recipe. Source: [DINOv2 notes](notes.md).
- **General-purpose visual features** — frozen or lightly adapted representations evaluated across domains and tasks. Source: [DINOv2 notes](notes.md).
