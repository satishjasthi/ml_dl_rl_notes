# SimCLR Questions

Source of truth: [notes.md](notes.md)

- Why does the projection head improve `h` even though it is discarded downstream?
- How does NT-Xent relate exactly to multiclass cross-entropy and InfoNCE?
- What changes when a minibatch contains semantically similar examples, creating false negatives?
- Which augmentations are safe for a particular downstream domain, especially satellite or geospatial imagery?
- What is the smallest practical prototype that demonstrates the positive-pair/negative-pair behavior without requiring ImageNet-scale training?
