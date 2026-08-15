# Claims

- **Strong augmentation composition is important for SimCLR.** Evidence: augmentation ablations reported in the paper. Source: [SimCLR notes](../papers/simCLR/notes.md).
- **A nonlinear projection head improves the encoder representation used downstream.** Evidence: projection-head ablations. Source: [SimCLR notes](../papers/simCLR/notes.md).
- **Interpretation:** SimCLR learns features stable under selected transformations by pulling positive views together and separating in-batch negatives. Source and qualification: [SimCLR notes](../papers/simCLR/notes.md).

- **Few-label semi-supervised learning can outperform the supervised baseline:** the SimCLRv2 paper reports 77.5% top-1 accuracy for ResNet-50 with 10% of labels, exceeding its cited standard supervised all-label baseline. Evidence: NeurIPS abstract and paper results; source [SimCLRv2 notes](../papers/simCLRv2/notes.md).
- **SimCLRv2's gains are workflow-level, not only loss-level:** the paper combines scaled contrastive pretraining, few-label fine-tuning, and unlabeled-data distillation. Evidence: paper abstract and method description; source [SimCLRv2 notes](../papers/simCLRv2/notes.md).
- **Interpretation:** self-supervised pretraining learns broad invariances, labels attach task semantics, and distillation spreads those semantics. This is an interpretation of the pipeline, not a separately isolated causal result. Source [SimCLRv2 notes](../papers/simCLRv2/notes.md).

- **MAE scalability claim:** high-ratio masking plus an asymmetric encoder-decoder can make visual self-supervised pretraining efficient while producing transferable representations. Evidence: the paper's ImageNet-1K experiments, masking ablations, and reported speedup; source [MAE notes](../papers/masked_auto_encoders/notes.md).
- **MAE high-masking claim:** approximately 75% masking is a meaningful and effective choice in the studied setup, rather than merely an efficiency trick. Evidence: paper ablations; source [MAE notes](../papers/masked_auto_encoders/notes.md).
- **Interpretation:** forcing reconstruction from sparse context encourages contextual representations, while the visible-only encoder reduces token computation. This is an explanatory interpretation, not a separately isolated causal finding. Source [MAE notes](../papers/masked_auto_encoders/notes.md).

- **DINO negative-free claim:** DINO learns transferable visual features using self-distillation without human labels or explicit contrastive negative examples. Evidence: method and ImageNet evaluation in the paper; source [DINO notes](../papers/dino-v1/notes.md).
- **DINO emergent-structure claim:** self-supervised Vision Transformers can produce semantically meaningful attention maps and object-discovery behavior in the studied experiments. Evidence: attention visualizations and object-discovery experiments; source [DINO notes](../papers/dino-v1/notes.md).
- **Interpretation:** EMA targets stabilize self-training while global-local crop agreement encourages information shared across views. This is a conceptual interpretation, not an independently isolated causal result. Source [DINO notes](../papers/dino-v1/notes.md).

- **DINOv2 general-purpose feature claim:** self-supervised visual features can transfer across image distributions and tasks when trained with sufficiently diverse curated data, model scale, and a strengthened recipe. Evidence: DINOv2 abstract and multi-task evaluations; source [DINOv2 notes](../papers/dino_v2/notes.md).
- **DINOv2 objective-combination claim:** global DINO-style distillation, iBOT-style masked patch prediction, and feature regularization complement one another in the studied training system. Evidence: paper method and ablations; source [DINOv2 notes](../papers/dino_v2/notes.md).
- **Interpretation:** global targets encourage semantic view invariance, patch targets preserve local context, and KoLeo-style regularization spreads features. This explains the design but is not a fully isolated causal attribution. Source [DINOv2 notes](../papers/dino_v2/notes.md).

- **Patch tokenization enables pure Transformer image recognition.** Evidence: ViT transfer experiments across ImageNet, CIFAR-100, VTAB, and other benchmarks; source [ViT notes](../papers/vit_paper/notes.md).
- **ViT benefits substantially from large-scale pretraining.** Evidence: the paper's comparison of pretraining/data scales and its smaller-data qualification; source [ViT notes](../papers/vit_paper/notes.md).
- **Removing CNN inductive bias is a conditional trade-off, not a universal CNN replacement claim.** Evidence: strong large-scale transfer results alongside weaker small-data-from-scratch behavior; source [ViT notes](../papers/vit_paper/notes.md).
- **Interpretation:** Smaller patches preserve more spatial detail but raise token count and global-attention cost. This explains an engineering trade-off, not an isolated causal result. Source [ViT notes](../papers/vit_paper/notes.md).
