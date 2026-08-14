# DINOv2 Resources

## Primary sources

- [DINOv2 paper — Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
  - Relationship: primary source
  - Relevance: Canonical paper for the data curation, training recipe, model scaling, objectives, ablations, and downstream evaluations.
- [DINOv2 paper PDF](https://arxiv.org/pdf/2304.07193)
  - Relationship: primary source
  - Relevance: Full equations, experimental tables, and implementation details.
- [Official DINOv2 implementation](https://github.com/facebookresearch/dinov2)
  - Relationship: implementation
  - Relevance: Author-provided training code, model definitions, pretrained checkpoints, and evaluation utilities.

## Related papers in this knowledge base

- [DINOv1 notes](../dino-v1/notes.md)
  - Relationship: predecessor and comparison
  - Relevance: Supplies the original EMA-teacher, multi-crop, centering, and sharpening intuition that DINOv2 extends.
- [MAE notes](../masked_auto_encoders/notes.md)
  - Relationship: comparison
  - Relevance: Contrasts DINOv2/iBOT semantic patch targets with MAE pixel reconstruction from visible patches.
- [SimCLRv2 notes](../simCLRv2/notes.md)
  - Relationship: comparison
  - Relevance: Contrasts negative-free teacher-student learning with contrastive pretraining and later task distillation.
