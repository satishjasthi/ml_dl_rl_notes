# MAE Concepts

- [ ] **Patch tokens** — split an image into non-overlapping regions and represent each region as a token. See [notes](notes.md#1-turn-an-image-into-patch-tokens).
- [ ] **Masked autoencoding** — learn from unlabeled data by reconstructing hidden input content. See [notes](notes.md#paper-at-a-glance).
- [ ] **Asymmetric encoder-decoder** — use a large encoder on visible tokens and a lightweight decoder on the reconstructed full sequence. See [notes](notes.md#method).
- [ ] **High masking ratio** — hide about 75% of patches so the task is difficult while reducing encoder computation. See [notes](notes.md#why-the-high-masking-ratio-helps).
- [ ] **Masked-patch reconstruction loss** — calculate pixel reconstruction error mainly on hidden patches. See [notes](notes.md#5-compute-reconstruction-loss-on-masked-patches).
- [ ] **Vision Transformer (ViT)** — Transformer architecture applied to image patch sequences. See [notes](notes.md#3-encode-only-visible-patches).
- [ ] **Contrastive learning versus masked prediction** — compare MAE's reconstruction target with SimCLRv2's view matching and distillation. See [notes](notes.md#comparison-with-simclrv2).
