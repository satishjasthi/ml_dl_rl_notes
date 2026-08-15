# Cross-paper Comparisons

## SimCLR versus SimCLRv2

| Dimension | [SimCLR](../papers/simCLR/notes.md) | [SimCLRv2](../papers/simCLRv2/notes.md) |
| --- | --- | --- |
| Primary objective | Learn representations from augmented positive pairs and in-batch negatives | Use scaled contrastive representations in a few-label semi-supervised pipeline |
| Core pretraining loss | NT-Xent / InfoNCE-style contrastive loss | SimCLR-style contrastive loss with larger models and improved scaling/configuration |
| Label usage | Primarily downstream evaluation and transfer | Fine-tune with few labels, then distill with unlabeled examples |
| Main scaling lesson | Large batches, long training, and strong augmentations matter | Large self-supervised models can be especially valuable in low-label regimes |
| Knowledge transfer | Representation transferred to downstream classifier | Task-specific teacher transfers soft predictions to a student |
| Best mental model | Learn invariance by matching two views | Learn broadly, specialize with few labels, then spread the specialization |
| Source | [SimCLR notes](../papers/simCLR/notes.md) | [SimCLRv2 notes](../papers/simCLRv2/notes.md) |

**Bottom line:** SimCLR is the representation-learning engine; SimCLRv2 packages that engine with scaling, few-label fine-tuning, and unlabeled-data distillation.

## MAE versus SimCLRv2

| Dimension | [MAE](../papers/masked_auto_encoders/notes.md) | [SimCLRv2](../papers/simCLRv2/notes.md) |
| --- | --- | --- |
| Pretext task | Reconstruct randomly masked image patches | Match augmented views with a contrastive objective |
| Encoder input during pretraining | Visible patches only | Full augmented views |
| Negative examples | Not required | In-batch negatives are central to the SimCLR-style stage |
| Temporary component | Lightweight reconstruction decoder | Projection head, then teacher/student heads in later stages |
| Label workflow | Downstream labels after pretraining | Few labels fine-tune a teacher; unlabeled data supports distillation |
| Efficiency idea | High masking ratio and asymmetric decoder | Scale the self-supervised model and reuse unlabeled data in a multi-stage pipeline |
| Source | [MAE notes](../papers/masked_auto_encoders/notes.md) | [SimCLRv2 notes](../papers/simCLRv2/notes.md) |

**Bottom line:** MAE learns by filling spatially missing content; SimCLRv2 learns by view agreement and then transfers task predictions. They share a self-supervised goal but use different learning signals and downstream workflows.

## DINO versus SimCLRv2 versus MAE

| Dimension | [DINO](../papers/dino-v1/notes.md) | [SimCLRv2](../papers/simCLRv2/notes.md) | [MAE](../papers/masked_auto_encoders/notes.md) |
| --- | --- | --- | --- |
| Core signal | Student matches EMA-teacher soft targets across crops | Match augmented positive views and separate in-batch negatives; later fine-tune and distill | Reconstruct masked image patches |
| Negative examples | Not required | Central during contrastive pretraining | Not required |
| Temporary component | EMA teacher and self-supervised head | Projection head, then task teacher/student heads | Lightweight reconstruction decoder |
| Main invariance/context pressure | Global-local crop agreement | Augmentation-induced instance agreement | Spatial context for missing-patch prediction |
| Best mental model | A slow reviewer teaches a fast learner to agree | Learn broadly, specialize with few labels, then spread task knowledge | Hide most of the image and fill in missing context |

**Bottom line:** DINO and SimCLRv2 both use view agreement, but DINO replaces explicit negatives with an EMA teacher and soft targets. MAE defines the pretext task as pixel reconstruction from visible patches. Source: [DINO notes](../papers/dino-v1/notes.md).


## DINOv1 versus DINOv2

| Dimension | [DINOv1](../papers/dino-v1/notes.md) | [DINOv2](../papers/dino_v2/notes.md) |
| --- | --- | --- |
| Core signal | EMA-teacher soft-target agreement across global and local views | Retains DINO-style global agreement and adds masked patch prediction |
| Patch-level learning | Not central to the original objective | iBOT-style student patch prediction from teacher targets |
| Anti-collapse/stability | Centering, sharpening, momentum, and multi-crop recipe | Retains those ideas and adds KoLeo-style feature spreading plus improved training practices |
| Data ambition | Strong self-supervised features and emergent ViT properties | General-purpose features from curated LVD-142M data and larger models |
| Main mental model | A slow reviewer teaches a learner to agree across crops | A scaled apprentice learns global semantics, local patch structure, and spread-out features |
| Source | [DINOv1 notes](../papers/dino-v1/notes.md) | [DINOv2 notes](../papers/dino_v2/notes.md) |

**Bottom line:** DINOv2 is best understood as a scaled and strengthened DINO family recipe, not a replacement of the original teacher-student intuition.

## ViT as the visual extension of the Transformer

| Dimension | [Attention Is All You Need](../papers/attention-is-all-you-need/notes.md) | [ViT](../papers/vit_paper/notes.md) |
| --- | --- | --- |
| Input tokens | Text/subword sequence | Flattened image patches plus a class token |
| Positional information | Sequence positional encodings | Learned patch-grid position embeddings |
| Core encoder | Self-attention and feed-forward blocks | Same standard Transformer encoder pattern |
| Task readout | Decoder or sequence outputs for translation | Final class-token representation for classification |
| Key challenge | Preserve order while modeling long-range language context | Learn visual structure that CNNs often provide as inductive bias |
| Scaling lesson | Parallel attention replaces recurrent processing | Large-scale pretraining makes a low-bias visual Transformer effective |

## ViT backbone versus later self-supervised ViT methods

| Dimension | [ViT](../papers/vit_paper/notes.md) | [MAE](../papers/masked_auto_encoders/notes.md) | [DINO](../papers/dino-v1/notes.md) |
| --- | --- | --- | --- |
| Backbone role | Supervised/pretrained image classifier | Encoder for masked image reconstruction pretraining | Student/teacher backbone for self-distillation |
| Learning signal | Labels in the paper's transfer setup | Reconstruct missing patches | Match EMA-teacher representations across crops |
| Shared foundation | Patch tokens, positional information, Transformer encoder | Builds on ViT-style patch processing | Builds on ViT-style patch processing |
| Main additional idea | Replace convolutional visual processing with global token attention | Learn from sparse visible context | Learn semantic invariance without labels or explicit negatives |
