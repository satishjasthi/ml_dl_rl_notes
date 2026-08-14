# Questions

- How do false negatives affect contrastive objectives, and what methods mitigate them? Raised by [SimCLR questions](../papers/simCLR/questions.md).
- Why does a projection head improve the pre-projection representation? Raised by [SimCLR questions](../papers/simCLR/questions.md).

- How much of SimCLRv2's low-label improvement comes independently from model scale, fine-tuning, and distillation? Raised by [SimCLRv2 questions](../papers/simCLRv2/questions.md).
- When do teacher soft targets amplify errors or fail under unlabeled distribution shift? Raised by [SimCLRv2 questions](../papers/simCLRv2/questions.md).

- Why does a 75% masking ratio work well, and how does this trade off reconstruction difficulty against encoder computation? Raised by [MAE questions](../papers/masked_auto_encoders/questions.md).
- How do MAE's spatial-context features differ from SimCLRv2's augmentation-induced invariances? Raised by [MAE questions](../papers/masked_auto_encoders/questions.md).

- Which DINO anti-collapse component contributes most under matched compute: centering, sharpening, teacher momentum, or multi-crop augmentation? Raised by [DINO questions](../papers/dino-v1/questions.md).
- How do DINO’s crop-and-teacher invariances differ experimentally from SimCLRv2’s contrastive invariances and MAE’s spatial-context features? Raised by [DINO questions](../papers/dino-v1/questions.md).

- Which DINOv2 components account for improvements under matched compute: data curation, iBOT patch prediction, KoLeo regularization, architecture, or training schedules? Raised by [DINOv2 questions](../papers/dino_v2/questions.md).
- How do DINOv2 semantic patch targets compare with MAE pixel reconstruction and DINOv1 global-only self-distillation? Raised by [DINOv2 questions](../papers/dino_v2/questions.md).
