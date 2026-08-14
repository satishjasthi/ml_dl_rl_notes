# Paper: A Simple Framework for Contrastive Learning of Visual Representations

- Authors: Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
- Published: arXiv 2020; ICML 2020
- Primary link: https://arxiv.org/abs/2002.05709
- PDF link: https://arxiv.org/pdf/2002.05709
- Code/data links: https://github.com/google-research/simclr
- Topics: self-supervised learning, contrastive learning, representation learning, computer vision
- Date started: 2026-08-13
- Study goals: Understand the intuition; Implement a prototype; Prepare for an interview
- Background: MLOps engineer
- Current priority: Build intuition first
- Understanding status: In progress
- Source provenance: Paper claims and experimental details are summarized from the supplied arXiv source, especially Sections 2–5, including the evaluation and ablation experiments. Interpretive explanations are labeled as intuition or interpretation.

## Paper at a glance

SimCLR learns useful image representations without human labels. For each image, it creates two differently augmented views and trains an encoder so that the two views have similar representations, while views from other images in the minibatch are less similar. After pretraining, the projection head is discarded and the encoder representation is used for downstream tasks.

The paper's main message is that a relatively simple recipe works very well when four ingredients are chosen carefully: strong compositions of augmentations, a trainable nonlinear projection head, a contrastive loss over many examples, and sufficiently large batches/long training. The method does not require a specialized architecture or memory bank.

## Intuition first

Imagine showing a model two crops of the same dog photo: one may be darker, zoomed, blurred, or differently colored. The pixels differ, but the semantic identity—"this depicts a dog"—should remain. SimCLR asks the model to keep what survives these transformations and ignore what does not.

For one image, the two augmented views form a **positive pair**. Every other view in the minibatch is treated as a **negative** for that anchor. The model is rewarded for placing the positive close in embedding space and penalized when a negative is closer than the positive. Repeating this across many images makes the representation organize images by features stable under the chosen augmentations.

A useful mental model is a continual sorting exercise: pull the two views of each underlying image together, while pushing apart views belonging to different underlying images. The augmentations define what the model is allowed to regard as the same; therefore, augmentation design is part of the task definition, not merely data preprocessing.

## Problem and motivation

Traditional supervised visual representations require many labeled examples. Earlier self-supervised contrastive methods often used specialized architectures or memory banks to obtain enough negative examples. SimCLR investigates how far a simpler setup can go: standard data augmentation, an encoder, a small projection MLP, and in-batch negatives.

## Method

1. Sample a minibatch of $N$ images.
2. Independently sample two augmentations from an augmentation family $T$ for each image, producing $2N$ views.
3. Encode each view with an encoder $f$, such as a ResNet, to obtain $h = f(\text{view})$.
4. Map $h$ through a nonlinear projection head $g$, typically an MLP, to obtain $z = g(h)$.
5. For each view, classify its paired view among the other $2N - 1$ views using the normalized temperature-scaled cross-entropy loss (NT-Xent).
6. After pretraining, use $h$—not $z$—for downstream evaluation; the projection head is discarded.

## Cross-entropy and NT-Xent in detail

### 1. The intuition behind cross-entropy

A classification model produces a probability distribution over possible answers. Cross-entropy measures how much probability the model assigned to the correct answer. For one training example:

$$
\text{loss} = -\log(\text{probability assigned to the correct answer})
$$

The negative logarithm gives the desired behavior:

- If the correct probability is close to $1$, the loss is close to $0$.
- If the correct probability is moderate, the loss is moderate.
- If the model is confidently wrong and assigns probability close to $0$ to the correct answer, the loss becomes very large.

For example:

| Probability assigned to the correct answer | Loss $-\log(p)$ |
| ---: | ---: |
| $0.99$ | approximately $0.010$ |
| $0.90$ | approximately $0.105$ |
| $0.50$ | approximately $0.693$ |
| $0.10$ | approximately $2.303$ |
| $0.01$ | approximately $4.605$ |

Cross-entropy is therefore also the negative log-likelihood of the correct target. Training minimizes it by increasing the probability assigned to the correct target.

### 2. Binary cross-entropy

In binary classification, the target is $y \in \{0,1\}$. The model produces a probability $p$ that the target is $1$.

The binary cross-entropy loss is:

$$
\mathcal{L}_{\text{BCE}}
= -\left[y\log(p) + (1-y)\log(1-p)\right]
$$

The two terms allow the same equation to handle both possible targets.

#### When the target is $y=1$

The equation becomes:

$$
\mathcal{L}_{\text{BCE}} = -\log(p)
$$

If the model predicts $p=0.9$, then:

$$
\mathcal{L}_{\text{BCE}} = -\log(0.9) \approx 0.105
$$

This is a small loss because the model gave high probability to the correct answer.

If the model predicts $p=0.01$, then:

$$
\mathcal{L}_{\text{BCE}} = -\log(0.01) \approx 4.605
$$

This is a large loss because the model was confidently wrong.

#### When the target is $y=0$

The equation becomes:

$$
\mathcal{L}_{\text{BCE}} = -\log(1-p)
$$

If the model predicts $p=0.1$, it assigns probability $0.9$ to the correct answer $y=0$:

$$
\mathcal{L}_{\text{BCE}} = -\log(0.9) \approx 0.105
$$

The model usually produces $p$ by applying a sigmoid to a scalar logit $a$:

$$
 p = \sigma(a) = \frac{1}{1+\exp(-a)}
$$

The logit is an unrestricted real-valued score; the sigmoid converts it into a number between $0$ and $1$.

### 3. Multiclass cross-entropy

In a $C$-class problem, the model produces one logit for each class:

$$
\mathbf{a} = [a_1,a_2,\ldots,a_C]
$$

The logits are converted into probabilities with softmax:

$$
 p_c = \frac{\exp(a_c)}{\sum_{r=1}^{C}\exp(a_r)}
$$

The probabilities are positive and sum to $1$.

If the correct class is $y$, multiclass cross-entropy is:

$$
\mathcal{L}_{\text{CE}} = -\log(p_y)
$$

Using a one-hot target vector $\boldsymbol{y}$, the same equation can be written as:

$$
\mathcal{L}_{\text{CE}}
= -\sum_{c=1}^{C} y_c\log(p_c)
$$

Only the correct class has $y_c=1$, so the sum reduces to $-\log(p_y)$.

#### Numerical example

Suppose the three class logits are:

$$
\mathbf{a} = [2,1,0]
$$

The softmax probabilities are approximately:

$$
\mathbf{p} = [0.665,0.245,0.090]
$$

If class 1 is correct, the loss is:

$$
\mathcal{L}_{\text{CE}} = -\log(0.665) \approx 0.408
$$

If class 3 is correct instead, the loss is:

$$
\mathcal{L}_{\text{CE}} = -\log(0.090) \approx 2.408
$$

The same prediction is penalized much more when the correct class receives low probability.

### 4. The bridge from ordinary classification to SimCLR

SimCLR also creates a multiclass classification problem, but its classes are not fixed semantic labels such as `cat`, `dog`, or `bottle`.

For an anchor view $i$:

- The matching view $j$ from the same original image is the correct target.
- Every other view in the minibatch is a candidate answer.
- Similarity between the anchor and each candidate acts as the candidate's logit.

Thus, the candidate set changes for every anchor and every minibatch.

| Ordinary multiclass classification | SimCLR NT-Xent |
| --- | --- |
| Fixed classes | Other views in the current minibatch |
| Correct class label | Matching augmented view |
| Class logits | Similarities between embeddings |
| Softmax over classes | Softmax over candidate views |
| Human-provided target | Automatically generated pair target |

This is why NT-Xent is best understood as a form of multiclass cross-entropy with dynamically generated classes.

### 5. SimCLR setup for NT-Xent

Start with a minibatch of $N$ original images. SimCLR independently samples two augmentations for every image, producing $2N$ views.

For each original image $k$, the two views form a positive pair:

$$
(2k-1,2k)
$$

For an anchor view $i$, let $j$ denote its paired view. All other views except the anchor itself are candidates:

$$
\{1,2,\ldots,2N\}\setminus\{i\}
$$

There are $2N-1$ candidates. Only one candidate, $j$, is the positive target for this anchor.

The projection head produces embeddings $z_i$ and $z_k$. SimCLR usually compares them using cosine similarity:

$$
\operatorname{sim}(z_i,z_k)
= \frac{z_i^{\mathsf{T}}z_k}{\lVert z_i\rVert\lVert z_k\rVert}
$$

Cosine similarity is high when the embeddings point in similar directions and low when their directions differ.

The similarity is divided by the temperature $\tau$:

$$
 s_{i,k} = \frac{\operatorname{sim}(z_i,z_k)}{\tau}
$$

These scaled similarities serve as the logits for the candidate-view classification problem.

### 6. Converting similarities into probabilities

The probability that candidate $k$ is the matching view for anchor $i$ is:

$$
 q_{i,k}
= \frac{\exp(s_{i,k})}
{\sum_{r\ne i}\exp(s_{i,r})}
$$

Substituting the similarity definition gives:

$$
 q_{i,k}
= \frac{\exp\left(\operatorname{sim}(z_i,z_k)/\tau\right)}
{\sum_{r\ne i}\exp\left(\operatorname{sim}(z_i,z_r)/\tau\right)}
$$

This is simply softmax over the similarities between the anchor and every other view.

### 7. The NT-Xent loss

Because $j$ is the correct candidate for anchor $i$, the cross-entropy loss is the negative log probability assigned to $j$:

$$
\ell(i,j) = -\log(q_{i,j})
$$

Substituting the softmax probability gives the SimCLR NT-Xent equation:

$$
\ell(i,j)
= -\log
\frac{\exp\left(\operatorname{sim}(z_i,z_j)/\tau\right)}
{\sum_{k\ne i}
\exp\left(\operatorname{sim}(z_i,z_k)/\tau\right)}
$$

The numerator is the positive-pair score. The denominator contains the positive and all negative candidates. The loss becomes small when the positive receives most of the probability mass.

SimCLR evaluates both directions of each positive pair. If views $i$ and $j$ came from the same image, it uses both:

$$
\ell(i,j)
\quad\text{and}\quad
\ell(j,i)
$$

The full minibatch objective is:

$$
\mathcal{L}
= \frac{1}{2N}
\sum_{k=1}^{N}
\left[
\ell(2k-1,2k) + \ell(2k,2k-1)
\right]
$$

### 8. Worked contrastive example

Suppose the batch contains three original images:

$$
A, B, C
$$

After two augmentations, there are six views:

$$
A_1,A_2,B_1,B_2,C_1,C_2
$$

For anchor $A_1$:

- Positive target: $A_2$.
- Negative candidates: $B_1$, $B_2$, $C_1$, and $C_2$.
- Total candidates: $5$, because the anchor itself is excluded.

Assume the softmax over the five candidate views produces:

| Candidate | Probability |
| --- | ---: |
| $A_2$ | $0.80$ |
| $B_1$ | $0.08$ |
| $B_2$ | $0.05$ |
| $C_1$ | $0.04$ |
| $C_2$ | $0.03$ |

The loss for anchor $A_1$ is:

$$
\ell(A_1,A_2) = -\log(0.80) \approx 0.223
$$

If the model instead assigns only $0.05$ probability to $A_2$, the loss becomes:

$$
\ell(A_1,A_2) = -\log(0.05) \approx 2.996
$$

The model is encouraged to increase the similarity of $A_1$ and $A_2$ and reduce the similarities to the other views.

### 9. What the gradient is trying to do

For a fixed anchor, let $q_{i,k}$ be the softmax probability for candidate $k$. The derivative of the cross-entropy loss with respect to a candidate score is:

$$
\frac{\partial \ell(i,j)}{\partial s_{i,k}}
= q_{i,k} - \mathbf{1}[k=j]
$$

where $\mathbf{1}[k=j]$ equals $1$ when $k$ is the positive candidate and $0$ otherwise.

For the positive candidate $j$:

$$
\frac{\partial \ell(i,j)}{\partial s_{i,j}}
= q_{i,j}-1
$$

This is negative whenever $q_{i,j}<1$, so gradient descent increases the positive score.

For a negative candidate $k\ne j$:

$$
\frac{\partial \ell(i,j)}{\partial s_{i,k}}
= q_{i,k}
$$

This is positive, so gradient descent decreases the negative score. A negative with a large probability receives a larger gradient and is treated as a harder negative.

Because the scores are produced from embedding similarities, these updates flow backward through the projection head and the encoder.

### 10. The role of temperature

The temperature controls how sharply similarities are converted into probabilities:

$$
 s_{i,k} = \frac{\operatorname{sim}(z_i,z_k)}{\tau}
$$

- A small $\tau$ magnifies differences between similarity scores and makes the softmax sharper.
- A large $\tau$ softens the distribution and spreads probability across more candidates.

A small temperature makes the model focus strongly on the most similar negatives. Temperature is therefore an important optimization and representation-quality hyperparameter.

### 11. Binary versus NT-Xent contrastive loss

SimCLR's NT-Xent is not a binary question such as:

> Are these two images a matching pair: yes or no?

Instead, for each anchor it asks a multiclass question:

> Which one of these $2N-1$ candidate views is the matching view?

A binary contrastive objective could independently classify each pair as positive or negative. SimCLR instead normalizes all candidates together with one softmax. This creates direct competition: increasing the probability of the positive necessarily reduces the probability available to the negatives.

The essential structure is:

$$
\text{anchor}
\longrightarrow
\text{similarity scores to candidates}
\longrightarrow
\text{softmax}
\longrightarrow
\text{cross-entropy with paired view as target}
$$

### 12. Why this objective can produce useful representations

The loss directly requires the model to identify an image despite changes introduced by augmentation. To succeed repeatedly, the encoder must preserve information shared by the two views and become less sensitive to augmentation-specific details.

This explanation is an interpretation of the mechanism. The paper's empirical evidence comes from its linear-evaluation results and ablations showing the importance of augmentation composition, the projection head, batch size, and training duration.

## Important design choices

- **Augmentations:** Random crop and resize, color distortion, and Gaussian blur are central in the paper's ablations. Composition of augmentations is more effective than relying on a single transformation.
- **Projection head:** The nonlinear head improves the quality of the representation $h$ learned by the encoder, even though the head itself is not retained for downstream tasks. Intuitively, it gives the loss a task-specific space in which augmentation-specific information can be removed without forcing $h$ to discard all useful information.
- **In-batch negatives:** Other examples in the minibatch provide negatives, avoiding a memory bank or special contrastive architecture.
- **Temperature and normalization:** Cosine similarity and temperature control how sharply the model distinguishes the positive from negatives.
- **Scale:** The reported results benefit from large batch sizes and longer pretraining, which increase the number of negatives and optimization opportunities.

## What the paper claims versus evidence

- **Authors' claim:** SimCLR substantially simplifies contrastive self-supervised learning while achieving strong visual representations. **Evidence:** ImageNet linear evaluation and transfer experiments reported in Section 4 and the ablations in Section 5 of the paper.
- **Authors' claim:** Augmentation composition, especially color distortion combined with cropping, is critical. **Evidence:** The augmentation ablations.
- **Authors' claim:** A nonlinear projection head improves the learned encoder representation. **Evidence:** Projection-head ablations comparing representations before and after the head.
- **Interpretation:** SimCLR works because it turns invariance into a learning signal: the model must preserve information shared by two transformed views and suppress transformation-specific details. This interpretation explains the results but is not itself a separately measured claim.

## Limitations and failure modes

- The learned invariances are dictated by the augmentations. An augmentation that changes task-relevant information can teach the model to ignore something important.
- Other images in the minibatch are assumed to be useful negatives. Semantically similar images can be false negatives, which may push related examples apart.
- The method is computationally demanding because performance improves with large batches and long training.
- The pretraining objective does not directly guarantee that every downstream task will benefit; the downstream task must align reasonably with the invariances induced during pretraining.

## Current understanding

- [x] High-level goal: learn visual features without labels by matching two views of the same image.
- [x] Positive and negative pair intuition.
- [x] Role of augmentations, encoder, projection head, and in-batch negatives.
- [ ] Derive NT-Xent step by step and connect it to cross-entropy.
- [ ] Understand why h is better for downstream tasks than z.
- [ ] Implement and smoke-test a small prototype.
- [ ] Prepare an interview-ready explanation and likely follow-up questions.

## Maximum likelihood and why it leads to cross-entropy

### 1. The central idea

**Maximum likelihood estimation (MLE)** chooses model parameters that make the observed data as probable as possible under the model.

Suppose the real world produces observations according to an unknown distribution $p_{\text{data}}$. We collect a dataset:

$$
\mathcal{D} = \{x_1,x_2,\ldots,x_n\}
$$

We choose a parameterized model $p_{\theta}$, where $\theta$ represents all learnable parameters. The model might be:

- a Bernoulli distribution for a binary outcome;
- a categorical distribution for a multiclass label;
- a Gaussian distribution for a continuous measurement;
- a neural network that predicts a conditional distribution such as $p_{\theta}(y\mid x)$.

The model is not usually trying to memorize the exact training examples. It is trying to learn parameter values that explain why those examples could have been generated.

A useful intuition is:

> Among all models available to us, choose the one under which the data we actually observed would be most plausible.

### 2. Likelihood versus probability

The same mathematical quantity can be viewed in two different ways.

If the parameters are fixed and the data vary, we usually call $p_{\theta}(\mathcal{D})$ a probability of the data.

If the observed data are fixed and we vary the parameters, we call the same expression a likelihood:

$$
\mathcal{L}(\theta;\mathcal{D})
= p_{\theta}(\mathcal{D})
$$

The semicolon emphasizes that $\theta$ is the variable being evaluated while the dataset is treated as observed and fixed.

MLE is therefore:

$$
\hat{\theta}_{\text{MLE}}
= \arg\max_{\theta} \mathcal{L}(\theta;\mathcal{D})
$$

This means: find the parameter value that maximizes the likelihood of the observed dataset.

Likelihood is not the probability of the parameters given the data. The Bayesian quantity $p(\theta\mid\mathcal{D})$ is a posterior distribution over parameters. MLE treats $\theta$ as an unknown fixed value to estimate; Bayesian inference places a probability distribution over possible parameter values.

### 3. A simple coin-flip example

Assume a coin has an unknown probability $\theta$ of landing heads. We observe ten flips with eight heads and two tails.

The probability of this particular ordered sequence under the model is:

$$
\mathcal{L}(\theta;\mathcal{D})
= \theta^8(1-\theta)^2
$$

If $\theta=0.5$, the model considers eight heads and two tails relatively unlikely. If $\theta=0.8$, the observed result is more plausible.

MLE chooses:

$$
\hat{\theta}
= \arg\max_{\theta\in[0,1]}
\theta^8(1-\theta)^2
$$

For this dataset, the maximizing value is:

$$
\hat{\theta}=\frac{8}{10}=0.8
$$

The model has inferred the pattern that heads appear approximately $80\%$ of the time.

This does not mean the next flip must be heads. It means that, among the Bernoulli models considered, $\theta=0.8$ best explains the observations.

### 4. Why log-likelihood is used

For independent observations, the likelihood factorizes:

$$
\mathcal{L}(\theta;\mathcal{D})
= \prod_{i=1}^{n}p_{\theta}(x_i)
$$

Products of many probabilities can become extremely small and are inconvenient for numerical optimization. Taking a logarithm solves both problems:

$$
\ell(\theta;\mathcal{D})
= \log\mathcal{L}(\theta;\mathcal{D})
= \sum_{i=1}^{n}\log p_{\theta}(x_i)
$$

The logarithm is strictly increasing, so maximizing likelihood and maximizing log-likelihood produce the same parameter estimate:

$$
\arg\max_{\theta}\mathcal{L}(\theta;\mathcal{D})
= \arg\max_{\theta}\log\mathcal{L}(\theta;\mathcal{D})
$$

Deep-learning libraries usually minimize losses rather than maximize objectives. Therefore, we minimize the negative log-likelihood (NLL):

$$
\mathcal{L}_{\text{NLL}}(\theta;\mathcal{D})
= -\sum_{i=1}^{n}\log p_{\theta}(x_i)
$$

Usually the average is used instead of the sum:

$$
\overline{\mathcal{L}}_{\text{NLL}}
= -\frac{1}{n}\sum_{i=1}^{n}\log p_{\theta}(x_i)
$$

The average changes the scale of the loss but not the minimizing parameter values for a fixed dataset.

### 5. Binary cross-entropy is Bernoulli negative log-likelihood

Consider supervised binary data:

$$
\mathcal{D} = \{(x_i,y_i)\}_{i=1}^{n},
\qquad y_i\in\{0,1\}
$$

A neural network maps $x_i$ to a logit $a_i$, and a sigmoid converts it into a probability:

$$
 p_i
= p_{\theta}(y_i=1\mid x_i)
= \sigma(a_i)
= \frac{1}{1+\exp(-a_i)}
$$

A Bernoulli model assigns the following probability to the observed label $y_i$:

$$
 p_{\theta}(y_i\mid x_i)
= p_i^{y_i}(1-p_i)^{1-y_i}
$$

For the entire independently sampled dataset, the conditional likelihood is:

$$
\mathcal{L}(\theta;\mathcal{D})
= \prod_{i=1}^{n}
 p_i^{y_i}(1-p_i)^{1-y_i}
$$

Taking the negative logarithm gives:

$$
-\log\mathcal{L}(\theta;\mathcal{D})
= -\sum_{i=1}^{n}
\left[
 y_i\log(p_i)
 + (1-y_i)\log(1-p_i)
\right]
$$

Dividing by $n$ produces the average binary cross-entropy:

$$
\mathcal{L}_{\text{BCE}}
= -\frac{1}{n}\sum_{i=1}^{n}
\left[
 y_i\log(p_i)
 + (1-y_i)\log(1-p_i)
\right]
$$

Therefore:

> Binary cross-entropy is the average negative log-likelihood of a Bernoulli conditional model.

This is not just an analogy. Under the Bernoulli modeling assumption, minimizing BCE is exactly maximum likelihood estimation implemented as minimization of negative log-likelihood.

### 6. Multiclass cross-entropy is categorical negative log-likelihood

Now suppose each target belongs to one of $C$ classes. The model produces logits $a_{i,1},\ldots,a_{i,C}$ for input $x_i$. Softmax defines a categorical probability distribution:

$$
 p_{i,c}
= p_{\theta}(y_i=c\mid x_i)
= \frac{\exp(a_{i,c})}
{\sum_{r=1}^{C}\exp(a_{i,r})}
$$

Represent the target with a one-hot vector $\boldsymbol{y}_i$, where $y_{i,c}=1$ for the observed class and $0$ otherwise. The probability assigned to the observed target is:

$$
 p_{\theta}(\boldsymbol{y}_i\mid x_i)
= \prod_{c=1}^{C}p_{i,c}^{y_{i,c}}
$$

For the dataset:

$$
\mathcal{L}(\theta;\mathcal{D})
= \prod_{i=1}^{n}\prod_{c=1}^{C}
 p_{i,c}^{y_{i,c}}
$$

The negative log-likelihood is:

$$
-\log\mathcal{L}(\theta;\mathcal{D})
= -\sum_{i=1}^{n}\sum_{c=1}^{C}
 y_{i,c}\log(p_{i,c})
$$

The average negative log-likelihood is exactly the usual multiclass cross-entropy:

$$
\mathcal{L}_{\text{CE}}
= -\frac{1}{n}\sum_{i=1}^{n}\sum_{c=1}^{C}
 y_{i,c}\log(p_{i,c})
$$

Because the target is one-hot, only the observed class contributes for each example:

$$
\mathcal{L}_{\text{CE}}
= -\frac{1}{n}\sum_{i=1}^{n}
\log p_{i,y_i}
$$

Therefore:

> Multiclass cross-entropy is the average negative log-likelihood of a categorical conditional model.

### 7. Cross-entropy as distribution matching

There is an even broader connection. Suppose $q$ is the target distribution and $p_{\theta}$ is the model distribution. Their cross-entropy is:

$$
H(q,p_{\theta})
= -\sum_x q(x)\log p_{\theta}(x)
$$

This can be decomposed into entropy plus Kullback–Leibler divergence:

$$
H(q,p_{\theta})
= H(q) + D_{\mathrm{KL}}(q\|p_{\theta})
$$

where:

$$
H(q) = -\sum_x q(x)\log q(x)
$$

and:

$$
D_{\mathrm{KL}}(q\|p_{\theta})
= \sum_x q(x)\log\frac{q(x)}{p_{\theta}(x)}
$$

The target entropy $H(q)$ does not depend on the model parameters $\theta$. Consequently:

$$
\arg\min_{\theta}H(q,p_{\theta})
= \arg\min_{\theta}D_{\mathrm{KL}}(q\|p_{\theta})
$$

So minimizing cross-entropy trains the model distribution to become close to the target distribution in the forward KL-divergence sense.

For a one-hot target, the target distribution has zero entropy. Cross-entropy then reduces to the negative log probability assigned to the observed answer.

### 8. How learning patterns from data fits into MLE

The data-generating distribution $p_{\text{data}}$ is unknown. We only observe finite samples:

$$
 x_1,x_2,\ldots,x_n \sim p_{\text{data}}
$$

We choose a model family $\{p_{\theta}:\theta\in\Theta\}$. MLE searches within this family for the member that best explains the observed sample:

$$
\hat{\theta}
= \arg\max_{\theta}\sum_{i=1}^{n}\log p_{\theta}(x_i)
$$

The learned parameters capture regularities that make the observations likely. For example, a classifier may learn that particular shapes, textures, or combinations of features make a label more likely.

Generalization requires assumptions. The most important are:

- **Representative data:** Training samples should reflect the situations encountered later.
- **Shared structure:** The examples should contain reusable patterns rather than being unrelated random events.
- **A suitable model family:** The model must be expressive enough to represent useful patterns.
- **Controlled complexity:** A model with unlimited freedom can memorize the training set instead of learning reusable structure.
- **Appropriate independence assumptions:** The likelihood factorization often assumes examples are independent and identically distributed, or at least conditionally independent under the chosen model.

MLE alone does not guarantee meaningful understanding. It defines an optimization principle. The data, model architecture, inductive biases, regularization, and evaluation task determine what kind of patterns are learned.

### 9. Connecting MLE to SimCLR

SimCLR does not attempt to model the full probability distribution of raw images, such as $p_{\theta}(x)$. Instead, for each anchor view $i$, it defines a conditional probability distribution over candidate views:

$$
q_{\theta}(k\mid i)
= \frac{\exp\left(\operatorname{sim}(z_i,z_k)/\tau\right)}
{\sum_{r\ne i}\exp\left(\operatorname{sim}(z_i,z_r)/\tau\right)}
$$

The target is the index $j$ of the augmented view that came from the same original image. The likelihood of the observed positive pairing is:

$$
\mathcal{L}_{\text{pair}}(\theta)
= \prod_{i}q_{\theta}(j(i)\mid i)
$$

Its negative log-likelihood is:

$$
-\log\mathcal{L}_{\text{pair}}(\theta)
= -\sum_i\log q_{\theta}(j(i)\mid i)
$$

This is exactly the sum of NT-Xent losses. Therefore, NT-Xent can be viewed as conditional maximum likelihood for the task:

> Given an anchor view and a set of candidate views, assign the highest probability to the candidate generated from the same original image.

The distinction is important:

- Ordinary supervised cross-entropy models $p_{\theta}(y\mid x)$ for human-defined labels.
- SimCLR models a conditional matching distribution $q_{\theta}(k\mid i)$ for automatically generated pair labels.

Both use the same negative-log-probability principle.

### 10. The complete conceptual chain

The relationship can be summarized as:

$$
\text{observed data}
\longrightarrow
\text{probability model}
\longrightarrow
\text{likelihood}
\longrightarrow
\text{log-likelihood}
\longrightarrow
\text{negative log-likelihood}
\longrightarrow
\text{cross-entropy}
$$

For SimCLR, the final model is a softmax over similarity scores:

$$
\text{augmented views}
\longrightarrow
\text{encoder and projection embeddings}
\longrightarrow
\text{similarities}
\longrightarrow
\text{softmax matching distribution}
\longrightarrow
\text{NT-Xent}
$$

The deep connection is therefore real: cross-entropy is not an arbitrary penalty chosen independently of statistics. It is the negative log-likelihood objective for common probabilistic output models, and SimCLR applies the same principle to a self-supervised matching distribution.

## Interaction log

## Interaction 1 — Session setup and intuition
**Q:** Can `papers/simCLR/notes.md` be the confirmed workspace for studying SimCLR?
**A:** Confirmed the active paper as arXiv:2002.05709, the notes path as `papers/simCLR/notes.md`, and the goals as intuition, implementation, and interview preparation, with intuition first. Created the standard paper workspace and initial linked knowledge entries. Explained the core intuition: two augmented views of one image are pulled together while views from other images are pushed apart; augmentations define the invariances being learned. No prototype was created because implementation has not yet been explicitly requested.

## Interaction 2 — Instance discrimination, projection head, and contrastive-loss math
**Q:** How does SimCLR learn class-relevant representations when same-class images are negatives, what happens for multi-object crops, what is the projection head, and how does NT-Xent relate to binary and multiclass cross-entropy?
**A:** Explained that SimCLR uses instance-level pseudo-labels: two views of one image are positive, while other minibatch views—including possible same-class images—are negatives; class separation is evaluated later with a supervised linear layer, and same-class negatives are a known false-negative limitation. Multi-object or non-overlapping crops remain a positive pair, so augmentation choice determines whether the learned invariance is useful. Clarified that the projection head is a train-time MLP applied to the encoder output, usually trained jointly rather than attached to a pretrained encoder, and is discarded downstream. Derived binary cross-entropy, multiclass softmax cross-entropy, and SimCLR's NT-Xent as multiclass cross-entropy over similarity scores with the paired view as the target.

## Interaction 3 — Detailed cross-entropy and NT-Xent notes
**Q:** Can the notes be updated with a detailed explanation of cross-entropy, NT-Xent, intuition, examples, and mathematical equations?
**A:** Replaced the compact loss summary with a detailed LaTeX-formatted tutorial covering the intuition and equations for binary cross-entropy, sigmoid, multiclass softmax cross-entropy, the mapping from cross-entropy to SimCLR's dynamically generated candidate classes, cosine similarity, temperature scaling, NT-Xent, the full bidirectional minibatch objective, worked examples, the gradient direction, and the distinction between binary and multiclass contrastive formulations.

## Interaction 4 — Maximum likelihood and cross-entropy
**Q:** How does maximum likelihood connect to cross-entropy, learning patterns from data, and statistical inference from a distribution?
**A:** Added a detailed explanation of MLE as choosing parameters that maximize the probability of observed data, distinguishing likelihood from probability and Bayesian posterior. Connected independent-sample likelihood to log-likelihood and negative log-likelihood, derived binary cross-entropy from the Bernoulli model and multiclass cross-entropy from the categorical model, explained cross-entropy as entropy plus KL divergence, discussed assumptions required for generalization, and showed that SimCLR's NT-Xent is conditional maximum likelihood over the matching-view distribution rather than a model of raw image likelihood.


## Interaction 5 — Commit and push request
**Q:** Can the SimCLR notes changes be committed and pushed to the remote branch?
**A:** The request was recorded, but Git command execution is not available in the current tool session, so commit and push could not be performed or verified. No commit or remote branch was claimed.

## Interaction 6 — Repeated commit and push request
**Q:** Can the SimCLR changes now be committed and pushed?
**A:** Git execution is still unavailable in this session, so no commit or push could be performed or verified. The user must run the previously provided `git add`, `git commit`, and `git push` commands in the repository shell.

## Interaction 7 — Paper-local uv environment and five-image prototype
**Q:** Can all Python work for this paper use a reproducible uv environment inside the paper folder?
**A:** Created `papers/simCLR/pyproject.toml`, `uv.lock`, `.gitignore`, and the paper-local `.venv` using Python 3.11 with pinned `torch==2.7.1`, `torchvision==0.22.1`, and `Pillow==11.3.0`. Added `prototypes/five_image_batch/`, which downloads five public images, creates two random augmented views per image, runs a randomly initialized ResNet-18 and two-layer projection head, and computes bidirectional NT-Xent with detailed prints. Validated with `uv run --locked`; the run succeeded on Apple MPS with ten views, `h` shape `(10, 512)`, `z` shape `(10, 128)`, and loss `2.182603`.

## Interaction 8 — Persisting augmented views
**Q:** Can the prototype save both generated augmented views for every source image in a folder named after that image?
**A:** Updated the prototype to denormalize and save `view_1.png` and `view_2.png` under `prototypes/five_image_batch/views/<image_name>/`, added the generated-view directory to `.gitignore`, and documented the layout in the prototype README. Validated with `uv run --locked`; all ten view files were saved for `image_0` through `image_4`, and the ResNet/projection/NT-Xent flow still completed successfully.
