# ViT questionnaire

Use the stages in order. Try answering each question before opening [answers.md](answers.md). The questions test the paper's ideas, not trivia.

## Stage 1 — Conceptual understanding

1. What is the core architectural move in ViT, and what CNN assumption does it remove?
2. An image has shape $224 \times 224 \times 3$ and the patch size is $16$. How many patch tokens are produced? How many tokens enter the Transformer after adding a class token?
3. Why is a linear patch projection needed if the patches already contain pixel values?
4. What information is lost if ViT uses patch tokens but no positional embeddings?
5. What is the role of the class token? Does it correspond to a particular image region?
6. Why can ViT underperform a CNN when trained from scratch on a small dataset?
7. If patch size is reduced from $16$ to $8$ at fixed image resolution, what happens qualitatively to sequence length, attention cost, and fine detail?
8. State the difference between the paper's claim that CNN reliance is not necessary and the stronger claim that CNNs are never useful.

## Stage 2 — Application-level questions

9. Design a ViT input pipeline for $256 \times 256$ RGB satellite tiles with patch size $16$. Give the patch count, token count including the class token, and two compute or memory concerns.
10. A pretrained ViT accepts $224 \times 224$ inputs, but your downstream task uses $384 \times 384$ images with the same patch size. What must happen to the position embeddings, and why?
11. You have only 10,000 labeled images but access to a large unlabeled image collection. Propose a training strategy based on the paper's lessons and justify whether you would use pure ViT, a CNN, or a hybrid.
12. A model's validation accuracy improves when changing from patch size 32 to 16, but GPU memory usage becomes too high. Give at least three practical responses and explain their trade-offs.
13. You need classify tiny objects occupying only a few pixels. Would you increase or decrease patch size? What additional architectural or data considerations matter?
14. A researcher claims that a ViT has learned object relationships because one attention map looks object-shaped. How would you evaluate this claim more carefully?
15. Sketch the tensor shapes for a toy model with batch size $B=8$, image size $32 \times 32$, patch size $4$, RGB input, and embedding width $D=64$.
16. Your ViT trains but achieves near-random accuracy, while a CNN on the same data succeeds. List a debugging order that distinguishes shape, optimization, data, positional, and inductive-bias problems.

## Stage 3 — Scenario-level complex questions

17. You must deploy a ViT for $1024 \times 1024$ imagery under a strict latency budget. Compare three redesigns: larger patches, local/window attention, and a hierarchical or hybrid encoder. What accuracy risks and tests would you attach to each choice?
18. A domain shift changes image resolution and sensor characteristics simultaneously. The pretrained ViT is strong on the source domain but unstable after fine-tuning. Diagnose likely causes and propose an adaptation plan that isolates them.
19. Two teams report different conclusions: Team A says ViT is superior to CNNs; Team B says it is inferior. Design a fair experiment matrix covering data scale, model scale, patch size, pretraining, augmentation, compute budget, and transfer protocol.
20. A medical-imaging task contains mostly local texture cues, rare small lesions, and only a few thousand labels. Make a reasoned architecture and training recommendation. Include when a pure ViT might still be preferable.
21. You inherit a ViT whose position embeddings were interpolated from a square grid to a non-square deployment resolution. The model's performance drops only on wide images. What would you inspect and how would you fix or test it?
22. Your production team proposes using the class-token attention map as an explanation for every prediction. Explain why this may be insufficient, and design a more defensible evaluation of explanations.
23. You have a fixed pretraining compute budget. Would you spend it on more data, a larger model, smaller patches, or longer training? Formulate a decision framework grounded in the paper's scaling message rather than choosing one universally.
24. Design an ablation study to determine whether a performance gain came from global attention, increased token count, extra parameters, or better pretraining. Specify controls and measurements.
