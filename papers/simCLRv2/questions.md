# SimCLRv2 Questions

Source of truth: [notes.md](notes.md)

- [ ] Which exact projection-head, optimizer, augmentation, and ResNet configurations account for each reported SimCLRv2 gain?
- [ ] How much of the low-label improvement comes from pretraining scale, fine-tuning, and distillation individually?
- [ ] When should the teacher's soft predictions be filtered or calibrated before distillation?
- [ ] How sensitive is the pipeline to false negatives and unlabeled examples outside the labeled distribution?
- [ ] How does the SimCLRv2 recipe compare with later momentum-encoder or non-contrastive methods under equal compute?

## Conceptual quiz — attempt before viewing answers

This quiz is designed for a deep-learning practitioner. Answer in your own words; equations are optional unless requested. The answer key is intentionally omitted so the quiz can be used interactively.

### Level 1 — Concept understanding

1. **Three-stage pipeline:** Explain the purpose of each SimCLRv2 stage: contrastive pretraining, few-label fine-tuning, and distillation. What kind of information is learned or added at each stage?
2. **SimCLR connection:** In the pretraining stage, what are the positive pair and the negatives? How does this differ from ordinary supervised classification?
3. **Labels versus unlabeled data:** Why can unlabeled data be useful in Stage 1 and Stage 3, while labels are especially important in Stage 2?
4. **Projection head:** Why is the projection head used during contrastive training but normally discarded for downstream tasks?
5. **Soft targets:** What information can a teacher's probability distribution provide that a one-hot hard label does not?
6. **Scaling:** Why might a larger model benefit more from self-supervised pretraining than from supervised training with very few labels?
7. **SimCLR versus SimCLRv2:** State the single most important scope difference between the two papers.

### Level 2 — Application and real-life scenarios

8. **Satellite crop monitoring:** You have 1% labeled satellite images for crop-type classification and a much larger unlabeled archive from the same region and season. Design the three SimCLRv2 stages. For each stage, name the data, model component, and training signal you would use.
9. **Edge deployment:** Your best fine-tuned teacher is too expensive to run on a field device, but you have many unlabeled images from the deployment region. Explain how SimCLRv2 would address this constraint and what the student should learn from the teacher.
10. **Choosing evaluation:** After contrastive pretraining, you obtain strong linear-probe accuracy but poor accuracy after few-label fine-tuning. What does this suggest you should investigate? Give at least two hypotheses or experiments.
11. **Augmentation design:** In a factory defect-detection task, color intensity is itself a defect signal. Which SimCLR-style augmentation would you treat cautiously, and why? How would you test whether the augmentation is teaching the wrong invariance?
12. **Resource planning:** You can afford either a much larger batch for a small encoder or a much larger encoder with a moderate batch. Based on the two papers, what trade-offs would you consider before choosing? Distinguish a paper-backed claim from your own engineering judgment.

### Level 3 — Limitation and failure-scenario analysis

13. **Distribution shift:** The labeled training images are daytime satellite images, but most unlabeled images used for distillation are nighttime images from a different sensor. Predict two ways the pipeline can fail and propose mitigations.
14. **False negatives:** In a bird-species dataset, many different images in the same contrastive batch show the same species. Explain how treating them as negatives can distort the representation and suggest an experiment or mitigation.
15. **Teacher overconfidence:** The few-label teacher has learned a spurious background correlation and is highly confident on unlabeled examples with that background. Why can distillation make the problem worse? Give two safeguards.
16. **Task-destructive augmentations:** A medical-imaging task depends on a small localized pattern, but random crops often remove it. Explain how Stage 1 may encode an undesirable invariance and how you would detect and correct it.
17. **Scale and compute:** A team reproduces the toy workflow with a tiny dataset and concludes that SimCLRv2 provides no benefit because the student is not better than the teacher. Identify at least three reasons this conclusion may not transfer from the paper's setting.
18. **Decision question:** Give one concrete situation in which you would prefer a simpler supervised or semi-supervised baseline over SimCLRv2. Justify the decision using data scale, label availability, compute, or risk of distribution shift.

### Suggested response protocol

Start with Questions 1–7. After grading, continue to Questions 8–12, then Questions 13–18. For each answer, separate: **paper claim**, **evidence or mechanism**, and **your engineering judgment**.
