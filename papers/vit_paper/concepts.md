# ViT concepts

- [ ] Patch tokenization — convert a 2-D image grid into a sequence of flattened, linearly embedded patches; central to Section 3.1 and the toy prototype.
- [ ] Patch-size versus attention-compute trade-off — smaller patches preserve detail but increase $N$ and approximately quadratic attention cost.
- [ ] Class token — a learned sequence element used as the image-level readout for classification.
- [ ] Learnable positional embeddings — restore patch-location information after flattening the image into a sequence.
- [ ] Inductive bias — understand why CNN locality helps small-data learning and what ViT must learn from pretraining.
- [ ] ViT scaling behavior — connect data scale, model scale, patch size, transfer learning, and compute.
- [ ] Position-embedding interpolation — adapt learned spatial embeddings when downstream image resolution changes.
- [ ] Hybrid CNN-Transformer — compare a convolutional feature-map stem with pure patch embedding.
- [ ] Global self-attention over image patches — reason about long-range interactions and quadratic cost.
