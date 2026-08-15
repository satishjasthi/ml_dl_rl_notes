# Paper: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

- Authors: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
- Published: arXiv 2020; ICLR 2021
- Primary link: https://arxiv.org/abs/2010.11929
- PDF link: https://arxiv.org/pdf/2010.11929
- HTML link: https://arxiv.org/html/2010.11929v2
- Code/data links: No implementation link recorded from the supplied primary source; the prototype in this workspace is an independent educational implementation.
- Topics: Vision Transformer, image classification, self-attention, image patches, transfer learning, scaling
- Date started: 2026-08-15
- Study goals:
  - Understand the intuition — in progress
  - Implement a prototype — in progress
  - Prepare for an interview/presentation — in progress
- Background: Familiar with deep learning
- Current priority: Build intuition first, then use the toy prototype and staged questionnaire for conceptual, application, and scenario-level interview practice.
- Understanding status: In progress
- Source provenance: Paper-specific facts are summarized from the supplied arXiv source, especially Sections 1, 3, 4, and 5. Prototype behavior is an educational implementation and is not evidence for the paper's benchmark claims.

## Paper at a glance

The paper asks whether image recognition really needs convolutional inductive biases. Its answer is a pure Transformer applied to a sequence of image patches. Vision Transformer (ViT) splits an image into fixed-size patches, treats each flattened patch as a token, adds positional information and a learnable class token, and processes the resulting sequence with a standard Transformer encoder. A classification head reads the final class-token representation.

The central result is conditional rather than universal: with sufficiently large-scale pretraining, ViT transfers very strongly to image-recognition benchmarks and can match or outperform strong convolutional networks while using less pretraining compute in the studied comparisons. The paper also shows that the same architecture is less effective when trained from scratch on smaller datasets, because it has weaker built-in visual inductive biases than a CNN.

## Intuition first

### The conceptual leap

A CNN sees an image through local filters whose weights are reused across locations. This gives locality and translation-equivariance almost for free. ViT instead says: **turn the image into a sequence, then let a Transformer learn relationships among the patches**.

Imagine cutting a photograph into a grid of small tiles. Each tile becomes one word-like token. Self-attention lets every tile ask which other tiles are relevant for recognizing the image. A patch containing a dog’s ear can directly interact with patches containing the face or body, even when they are far apart in the grid. The model must learn visual locality and other spatial regularities from data rather than receiving them as hard-coded CNN structure.

### Why patches instead of pixels?

Self-attention compares every token with every other token, so its cost grows roughly quadratically with sequence length. Treating every pixel as a token would create a very long sequence. A patch of size $P \times P$ reduces an image with height $H$ and width $W$ to

$$
N = \frac{HW}{P^2}
$$

patch tokens, assuming $P$ divides both dimensions. Increasing $P$ shortens the sequence and reduces attention cost, but also discards fine spatial detail. Thus patch size is both a representation choice and a compute/accuracy trade-off.

### What happens to one image?

1. **Patchify:** reshape the image into a grid of non-overlapping patches.
2. **Embed:** flatten each patch and project it with a learned linear layer into the Transformer width.
3. **Add a class token:** prepend a learned token whose final representation will summarize the image for classification.
4. **Add position embeddings:** tell the model where each patch came from; otherwise the token sequence would not preserve the 2-D arrangement.
5. **Run a Transformer encoder:** each patch can exchange information with all other patches through self-attention and feed-forward layers.
6. **Classify:** apply a small head to the final class-token representation.

The class token is a learned “readout slot.” It does not correspond to a physical image patch; it gathers information through attention and becomes the input to the classifier.

## Method and equations

### Patch embedding and token sequence

For an image with $C$ channels and patch size $P$, flatten patch $i$ into

$$
 x_p^i \in \mathbb{R}^{P^2 C}.
$$

A learned projection $E \in \mathbb{R}^{(P^2C) \times D}$ maps it to the Transformer width $D$. The input sequence is

$$
 z_0 = \left[ x_{\mathrm{class}};\; x_p^1E;\; x_p^2E;\; \ldots;\; x_p^N E \right] + E_{\mathrm{pos}},
$$

where $x_{\mathrm{class}} \in \mathbb{R}^{D}$ is the learnable class token and $E_{\mathrm{pos}}$ contains learnable position embeddings for the $N+1$ sequence positions. This is the paper's Equation (1) in Section 3.1, with notation lightly normalized for readability.

### Transformer encoder

The encoder applies LayerNorm, multi-head self-attention (MSA), residual connections, and a feed-forward multilayer perceptron (MLP):

$$
\begin{aligned}
 z'_\ell &= \operatorname{MSA}(\operatorname{LN}(z_{\ell-1})) + z_{\ell-1}, \\
 z_\ell &= \operatorname{MLP}(\operatorname{LN}(z'_\ell)) + z'_\ell.
\end{aligned}
$$

The final class-token vector is sent to a classification head. The paper uses the standard Transformer encoder pattern rather than introducing a vision-specific attention operation (Section 3.1).

For one attention head, the underlying information-routing operation is

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

Each patch produces a query, key, and value. The query-key scores decide which patches matter to a destination token; the values carry the information that is aggregated. Multi-head attention repeats this in several learned subspaces.

### Fine-tuning and resolution changes

The paper pretrains ViT and then transfers it to downstream image-recognition tasks. During fine-tuning, the classification head is replaced for the target label set. When the input resolution changes, the number of patch tokens changes. The paper interpolates the pretrained position embeddings in 2-D rather than discarding them (Section 3.2). This is an important practical detail: patch size and image resolution jointly determine sequence length.

### Hybrid architecture

The paper also studies a hybrid in which a CNN feature map supplies the patch sequence before the Transformer. This tests whether a convolutional stem can provide useful low-level visual processing while retaining Transformer-style global interaction (Section 3.3). The main ViT result, however, is the pure patch-to-Transformer path.

## Why the idea matters

CNNs encode useful assumptions about images: nearby pixels interact, patterns can be reused across positions, and translation changes should often preserve recognition. These assumptions improve data efficiency, especially when data is limited. ViT removes most of those assumptions, making the architecture more uniform with the Transformer family and more amenable to large-scale pretraining. The trade-off is that ViT generally needs more data or stronger pretraining to learn visual structure reliably.

A useful interview summary is: **ViT trades hand-designed spatial inductive bias for a simple, scalable token-and-attention architecture. Large-scale pretraining supplies the data needed to learn the missing visual structure.**

## Training and experiments

The paper evaluates pretraining and transfer across large and smaller image-recognition datasets, including ImageNet, ImageNet-21k, JFT-300M, CIFAR-100, and VTAB (Section 4). It compares ViT model sizes and patch sizes with strong convolutional baselines and studies data scale, model scale, patch size, positional embeddings, and hybrid variants.

The main experimental pattern is more important than memorizing one score:

- On sufficiently large pretraining data, larger ViTs transfer extremely well.
- The benefit of scaling is stronger for ViT than for the compared ResNet baselines in the studied regime.
- On smaller datasets without strong pretraining, ViT can underperform CNNs.
- Smaller patches increase the token count and computation but can improve recognition by retaining more spatial detail.
- Position embeddings are necessary for the model to use spatial arrangement; the paper examines 1-D and 2-D interpolation when transferring resolutions.

These results support the authors' claim that CNN reliance is not necessary at scale, not the stronger claim that convolution is never useful.

## Paper claims, evidence, and interpretation

- **Authors' claim:** A pure Transformer applied to image-patch sequences can perform very well for image classification (abstract and Section 1).
- **Evidence:** Transfer experiments across multiple recognition benchmarks and ablations over data/model/patch scale (Section 4).
- **Authors' qualification:** ViT performs best with large-scale pretraining; on smaller datasets, its weaker inductive bias can hurt (Sections 1 and 4).
- **Interpretation:** Patchification is a tokenization interface, not a claim that image patches are semantically equivalent to words. The Transformer learns the useful relationships among those tokens.
- **Interpretation:** The class token is a learned aggregation mechanism. It is convenient, but mean pooling or other readouts can also be designed; the paper's specific architecture should not be generalized into “class tokens are always necessary.”

## Limitations and failure modes

1. **Data hunger:** Without enough data or pretraining, the model may not learn locality, edge structure, and translation-related regularities as efficiently as a CNN.
2. **Quadratic attention cost:** Increasing resolution or decreasing patch size increases $N$, and self-attention scales approximately as $O(N^2D)$ for sequence length $N$ and hidden width $D$.
3. **Patch information loss:** Large patches reduce compute but can erase small-object or boundary details.
4. **Position-transfer sensitivity:** Changing resolution requires careful position-embedding handling; naive resizing or incompatible token layouts can degrade performance.
5. **Benchmark and compute dependence:** The reported advantage depends on pretraining data, optimization, model scale, hardware, and comparison protocol. A toy implementation cannot validate the paper's large-scale claims.
6. **Global attention is not automatically superior:** A task with limited data, strict latency, or very high resolution may favor hierarchical, local, sparse, or hybrid architectures.

## Prototype and staged interview practice

The runnable prototype is documented in [prototypes/toy_vit/README.md](prototypes/toy_vit/README.md). It implements a tiny supervised ViT on synthetic $8 \times 8$ stripe images. It is deliberately faithful to the data path—patchify, linear patch embedding, class token, positional embedding, Transformer encoder, classification head—but is not a reproduction of the paper's pretraining or benchmark experiments.

The question bank is in [questions.md](questions.md), with answer explanations in [answers.md](answers.md). It progresses in the requested order:

1. conceptual understanding;
2. application-level design and debugging;
3. complex scenario-level reasoning.

## Current understanding

The current mental model is that ViT converts the spatial image problem into a sequence-modeling problem. Patch size controls the sequence length; the patch projection creates token features; position embeddings restore spatial identity; self-attention builds global patch relationships; and the class token provides a trainable readout. The decisive practical condition is scale: removing CNN inductive bias is viable when data and compute are sufficient to learn visual regularities.

## Interaction log




## Interaction 2 — Inductive bias and CNN trade-offs
**Q:** What does inductive bias mean intuitively, and why might CNNs suffer from it?
**A:** **Inductive bias, intuitively:** An inductive bias is a model's built-in assumption about what kinds of patterns are likely to be useful. It is not necessarily a flaw or statistical bias; it lets a model generalize from finite data instead of treating every possible function as equally plausible.

Imagine two students learning to recognize objects. One assumes that nearby pixels usually form meaningful patterns such as edges and textures. The other makes no such assumption and must learn everything from data. The first student may learn faster with fewer examples, while the second may eventually represent relationships that the first student's assumptions make harder to express.

CNNs encode several visual assumptions:

- **Locality:** nearby pixels should interact first.
- **Weight sharing:** the same pattern detector can be useful at different image locations.
- **Translation equivariance:** shifting an input tends to shift intermediate feature maps rather than requiring a completely new detector.
- **Hierarchical composition:** simple local patterns combine into larger shapes through successive layers.

These assumptions are usually helpful for natural images. For example, a CNN can detect a vertical edge or fur texture wherever it appears, then combine local patterns into parts and objects. This improves data efficiency.

The same assumptions can become restrictive when a task depends on exact location, long-range relationships, unusual spatial layouts, or a domain where local visual regularities differ from ordinary images. For example, if classification depends on the relationship between two distant objects, a CNN initially processes them separately and may need additional depth or pooling before they can interact. If a specific absolute location matters, translation-related behavior may also be less suitable by default.

So CNNs do not simply “suffer from inductive bias.” They benefit when the bias matches the task and suffer when it is mismatched. CNNs can learn around these limitations with depth, padding, pooling, coordinates, or global layers, but they may need additional capacity or data.

**CNN versus ViT:** A CNN assumes that nearby pixels matter first and that visual patterns can be reused across locations. A ViT divides an image into patches and lets self-attention learn which patches should interact. ViT has a weaker vision-specific bias, so it is more flexible but must learn more visual structure from data. This is why the ViT paper reports a trade-off: CNNs can be stronger with smaller datasets, while ViT becomes highly effective with sufficiently large-scale pretraining (Sections 1 and 4).

**Interview summary:** Inductive bias is a model's built-in assumption about useful patterns. CNNs assume locality, weight sharing, and translation-related structure, which makes them data-efficient for natural images but can restrict them when global relationships or absolute position matter. ViTs use weaker visual inductive biases, making them more flexible but more dependent on large-scale pretraining.


## Inductive bias in machine learning generally

The most important word is **inductive**.

### What does “inductive” mean?

Induction means trying to infer a general rule from a limited number of examples.

Suppose you observe:

- Example 1: when $x=1$, the output is $y=2$.
- Example 2: when $x=2$, the output is $y=4$.
- Example 3: when $x=3$, the output is $y=6$.

You may infer the rule $y=2x$ and predict that when $x=4$, the output will be $8$. But the three examples do not logically prove that rule. Infinitely many other rules could match those three observations and produce a different answer at $x=4$.

For example, one rule might be a straight line. Another rule might pass through all three points but curve sharply after $x=3$. The training examples alone cannot tell us which rule is correct.

A learning algorithm therefore needs some preference about which possible rules are more plausible. That preference is its **inductive bias**.

So, in general:

> **Inductive bias is the set of assumptions or preferences that helps a model choose how to generalize from observed examples to unseen examples.**

It is called a “bias” because the model is not treating all possible explanations equally. It is called “inductive” because the model is making a generalization beyond the examples it directly observed.

### Why is inductive bias unavoidable?

A finite dataset never uniquely determines the correct function. Even if a training set contains millions of examples, there can still be many functions that fit those examples.

A model must therefore answer questions such as:

- Should similar inputs receive similar outputs?
- Should the learned function be smooth?
- Should the relationship be linear or nonlinear?
- Should nearby image pixels interact first?
- Should the order of tokens matter?
- Should the same pattern be recognized in different locations?
- Should the output remain unchanged when the input is rotated, translated, or slightly perturbed?
- Should features be sparse, simple, low-dimensional, or hierarchical?

The answers to these questions determine what the model finds easy to learn and what it finds difficult. That is inductive bias.

Without any useful bias, a model could memorize the training set but have no principled reason to behave correctly on a new example. With a useful bias, the model can generalize using fewer examples. With a wrong or overly restrictive bias, the model may consistently make the wrong kind of generalization.

### A simple everyday analogy

Imagine seeing three kinds of birds for the first time. You observe that all of them have wings, feathers, and two legs. When you see a fourth unfamiliar bird, you infer that it probably also has wings, feathers, and two legs.

That inference is not guaranteed by the three observations. It is based on an assumption that nature has some regularity and that similar-looking biological structures will continue to occur in new examples.

A model does something similar, but its assumptions are determined by its representation, architecture, objective, training procedure, and data.

### Inductive bias is not the same as a bias term

In a neural-network equation such as

$$
 y = Wx + b,
$$

$b$ is called a bias parameter. That is a numerical offset learned by the model. It is not what we mean by **inductive bias**.

Inductive bias refers to the broader assumptions that shape generalization. It can arise from the architecture, the loss function, regularization, the optimizer, the initialization, the training data, or even the preprocessing pipeline.

It is also different from social or fairness bias. In ML discussions, “inductive bias” is usually a technical term about generalization assumptions, not a statement that a model is unfair or prejudiced.

### Examples across common ML models

#### Linear regression

A linear model assumes that the target can be approximated by a weighted combination of input features:

$$
\hat{y} = w_1x_1 + w_2x_2 + \cdots + w_dx_d + b.
$$

Its inductive bias is that the relationship is approximately linear and additive in the chosen features. This bias is useful when the real relationship is close to linear. It is restrictive when the target depends on complex interactions or highly nonlinear boundaries.

For example, a linear classifier can separate two classes with a hyperplane. It will find that kind of solution easier than a solution requiring many disconnected regions or intricate curves.

#### Polynomial regression and model capacity

A high-degree polynomial can fit a complicated curve through many points. That gives it more flexibility, but it may create unnecessary oscillations between the observed examples.

A low-degree polynomial has a stronger simplicity bias. It prefers smoother, simpler relationships. A high-degree polynomial has a weaker simplicity restriction and can fit more possibilities, but it becomes more vulnerable to overfitting.

This illustrates that inductive bias is connected to, but not identical to, model capacity. Capacity describes how many functions a model can represent; inductive bias describes which functions are preferred or easier to obtain.

#### k-nearest neighbors

A k-nearest-neighbor model assumes that nearby points in the input space tend to have similar outputs. To classify a new point, it looks at nearby training examples.

Its inductive bias is a form of local smoothness:

> Inputs that are close according to the chosen distance metric should usually have similar labels or targets.

This works well when the distance metric reflects meaningful similarity. It fails when two points are numerically close but semantically different, or when important relationships are not captured by that distance.

#### Decision trees

A standard decision tree prefers to solve a problem through a sequence of feature-based splits, such as:

- Is feature 1 less than a threshold?
- If yes, is feature 3 greater than another threshold?
- Continue recursively.

This gives a bias toward hierarchical, piecewise, often axis-aligned decision rules. Trees can represent complex functions, but they do not naturally prefer smooth changes in the output. Small changes near a split may cause abrupt changes in the prediction.

#### Naive Bayes

Naive Bayes assumes that features are conditionally independent given the class. This assumption is often false in a literal sense, but it can still produce useful predictions.

Its bias makes the estimation problem simpler: instead of learning every possible interaction among features, it learns separate feature contributions under the conditional-independence assumption.

### Examples across deep-learning models

#### CNNs

CNNs prefer local interactions, reusable features, and translation-related structure. These assumptions are well matched to many natural-image tasks, so CNNs can generalize effectively with less data.

#### RNNs

Recurrent neural networks process sequences step by step and maintain a hidden state. Their structure encodes a preference that the sequence order matters and that information can be carried through a compact evolving state.

This is useful for sequential data, but long-range information must pass through recurrent steps, which can make very distant dependencies difficult to learn.

#### Transformers

A Transformer allows tokens to interact through content-dependent attention. Its architecture makes direct long-range interactions easy, but it does not automatically know order. Positional encodings are therefore needed when order matters.

A Transformer has a different inductive bias from an RNN: it prefers flexible all-to-all token interactions rather than sequential state updates. A Vision Transformer applies this idea to image patches.

#### CNNs versus ViTs

A CNN starts with stronger assumptions about images. A ViT starts with a more general patch-token interface and global attention. This does not mean ViT has no inductive bias. ViT still has biases from:

- the choice of patch size;
- the use of positional embeddings;
- the class-token readout;
- the Transformer architecture;
- the optimization objective;
- data augmentation and pretraining data.

Its visual bias is simply weaker in some important respects than a CNN's local convolutional bias. Therefore ViT has more flexibility but often needs more data to learn useful visual structure.

#### Graph neural networks

A graph neural network generally assumes that information should be exchanged along graph edges and that the identity of node ordering should not matter. This is a strong and useful bias when the graph correctly describes relationships, such as molecular bonds or social connections.

If important relationships are missing from the graph, the model may struggle because its message-passing structure does not make those interactions easy.

### Inductive bias can come from more than the architecture

It is useful to think of inductive bias as a property of the entire learning system, not just the model class.

#### Architecture

Convolutions, recurrence, attention, pooling, graph message passing, and hierarchical layers all make some patterns easier than others.

#### Loss function

Mean-squared error encourages accurate numerical reconstruction. Cross-entropy encourages class separation. Contrastive losses encourage selected examples to be close or far apart. Masked reconstruction encourages a model to infer missing content from visible context.

The objective determines what kind of behavior is rewarded.

#### Regularization

Weight decay, dropout, early stopping, sparsity penalties, and smoothness constraints express preferences for certain solutions over others. For example, weight decay discourages very large parameter values, which often produces a preference for simpler or less extreme functions.

#### Data augmentation

If training images are randomly cropped, flipped, or color-jittered and the label is kept unchanged, the training procedure tells the model that those transformations should not change the class. Augmentation therefore creates an invariance bias.

If an image is rotated but its label is kept unchanged, the model is encouraged to treat rotation as irrelevant. That is useful only if rotation truly should not change the task label.

#### Pretraining data

A model pretrained on natural photographs develops different useful representations from a model pretrained on satellite imagery, medical scans, or text. The data distribution teaches the model what regularities are common and therefore creates an empirical inductive bias.

#### Optimization and initialization

Two models with the same architecture can represent the same mathematical functions, yet gradient-based training may find some functions more easily than others. Initialization, optimizer choice, learning-rate schedule, and training duration therefore influence the effective inductive bias.

### How inductive bias relates to overfitting and underfitting

Inductive bias creates a trade-off.

- **Too little or poorly aligned bias:** the model can fit many explanations, including memorization. It may perform very well on training data but poorly on unseen data.
- **Too much or mismatched bias:** the model is prevented from representing the true relationship and underfits or makes systematic errors.
- **Useful bias:** the model prefers explanations that match the actual structure of the task and therefore generalizes well.

This is one way to understand the bias-variance trade-off, although “inductive bias” is broader than the statistical bias term used in that formal decomposition.

A useful mental picture is a search space of possible functions. The dataset provides constraints, but many functions may remain possible. Inductive bias shapes the search space or the route through it:

- it may remove some functions entirely;
- it may make some functions easier to represent;
- it may make some functions more likely to be found during training;
- it may make some transformations produce similar representations.

### A practical way to identify a model's inductive bias

For any ML or DL model, ask:

1. **What relationships does the model make easy?**
2. **What relationships does it make difficult?**
3. **What transformations does it naturally treat as similar or equivalent?**
4. **What information does it preserve, summarize, or discard?**
5. **What assumptions are introduced by the loss, augmentation, and data?**
6. **What kinds of errors would be systematic if those assumptions were wrong?**

For ViT, the answers include patch-level representation, positional structure, and direct attention between tokens. For a CNN, they include local reusable patterns and translation-related behavior. For k-NN, they include local smoothness under a distance metric. For a linear model, they include approximately linear and additive relationships.

### Final intuition

Inductive bias is the model's **starting worldview** about how the world works.

The model does not learn from a blank slate. It comes with a structure that says, implicitly:

> “These kinds of explanations are natural, these relationships are easy, and these changes probably should or should not matter.”

Learning then adjusts that worldview using data.

A good inductive bias is not one that is weak in every situation. It is one that matches the structure of the task. CNNs are strong when local visual structure is useful. k-NN is strong when nearby examples are similar. Linear regression is strong when relationships are approximately linear. Transformers are strong when flexible long-range interactions matter and sufficient data can teach the model the relevant structure.

The ViT paper is an example of deliberately weakening one kind of inductive bias—the CNN's built-in visual bias—in exchange for a more general architecture that can benefit greatly from scale.

## Interaction 3 — Inductive bias in ML and DL generally
**Q:** What does “inductive” mean in inductive bias, and what does inductive bias mean across machine-learning and deep-learning models in general, not only CNNs?
**A:** Explained induction as generalizing a rule from finite observed examples to unseen cases, and inductive bias as the assumptions or preferences that guide that generalization. Added a full notes section covering linear models, polynomial models, k-nearest neighbors, decision trees, Naive Bayes, CNNs, RNNs, Transformers, GNNs, losses, regularization, augmentation, pretraining, optimization, overfitting, underfitting, and a practical checklist for identifying a model's bias. Clarified that inductive bias is broader than a neural-network bias parameter and is distinct from fairness bias. No prototype or code changes were made.


## Original Transformer encoder–decoder architecture in detail

The original Transformer from *Attention Is All You Need* is an **encoder–decoder** model for sequence-to-sequence tasks such as machine translation. For example, it can map an English sentence to a German sentence.

A useful high-level picture is:

$$
\text{source sequence}
\xrightarrow{\text{encoder}}
\text{contextual source representations}
\xrightarrow{\text{decoder with autoregressive generation}}
\text{target sequence}
$$

The encoder reads and represents the complete source sentence. The decoder uses those source representations together with the target tokens generated so far to produce the next target token.

The user's mental model is close, but three corrections are important:

1. The encoder does not normally end in one linear layer that creates one single latent vector. It produces a contextual representation for **every source-token position**.
2. The decoder is not a stack containing only self-attention followed by a linear layer. Each decoder block has masked self-attention, encoder–decoder cross-attention, and a position-wise feed-forward network, together with residual connections and layer normalization.
3. The decoder processes one token at a time only during autoregressive inference. During training, the entire target sequence is supplied at once and a causal mask prevents each position from seeing future target tokens.

### A translation example

Suppose the source sentence is:

> I love cats.

The target sentence might be:

> J'aime les chats.

Let the source length be $S$ and the target length be $T$. The source and target are tokenized, so the model may actually see subword tokens rather than complete words.

The source sequence is something like

$$
[\text{I},\ \text{love},\ \text{cats},\ \text{<EOS>}].
$$

The target sequence used for teacher forcing is shifted into two sequences:

- decoder input: $[\text{<BOS>},\ \text{J'aime},\ \text{les},\ \text{chats}]$;
- training target: $[\text{J'aime},\ \text{les},\ \text{chats},\ \text{<EOS>}].$

The decoder input at position $t$ is used to predict the target token at position $t$. The decoder does not receive the target answer at the same position as an input; it receives the previous target tokens.

## Part 1: The encoder

### Step 1: Source tokens become vectors

Each source token is converted into an embedding of width $d_{\text{model}}$. Positional information is added because self-attention alone does not know token order.

If the source length is $S$, the initial encoder representation has shape

$$
X^{(0)} \in \mathbb{R}^{S \times d_{\text{model}}}.
$$

For a batch of size $B$, the shape is

$$
X^{(0)} \in \mathbb{R}^{B \times S \times d_{\text{model}}}.
$$

Every row corresponds to one source-token position.

### Step 2: One encoder block

Each encoder block contains two major sublayers:

1. multi-head self-attention;
2. a position-wise feed-forward network.

Residual connections and layer normalization surround these sublayers. In the original Transformer, the conceptual pattern is

$$
\begin{aligned}
\widetilde{X}^{(\ell)} &= \operatorname{LayerNorm}\left(X^{(\ell-1)} + \operatorname{MHA}_{\text{self}}(X^{(\ell-1)})\right), \\
X^{(\ell)} &= \operatorname{LayerNorm}\left(\widetilde{X}^{(\ell)} + \operatorname{FFN}(\widetilde{X}^{(\ell)})\right).
\end{aligned}
$$

The exact placement of LayerNorm differs in some modern Transformer implementations, but the original paper uses the residual-plus-normalization structure described in Section 3.1.

### Encoder self-attention

For encoder self-attention, queries, keys, and values all come from the source sequence:

$$
Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V.
$$

Each source position can attend to every source position. For example, the representation of “cats” can use information from “I” and “love”. The representation at each position becomes contextual rather than being only the original word embedding.

For one head:

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

If the source length is $S$, the attention-score matrix has shape

$$
QK^\top \in \mathbb{R}^{S \times S}.
$$

There is no causal mask in the encoder self-attention of the original translation Transformer. A source token is allowed to use information from tokens both before and after it because the complete source sentence is available.

### Position-wise feed-forward network

After self-attention, each source position independently passes through the same feed-forward network:

$$
\operatorname{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2.
$$

The same weights are applied to every sequence position, but each position has a different contextual vector as input. The feed-forward network mixes features within one token representation; the attention sublayer mixes information across token positions.

### What does the encoder output?

After the final encoder block, the encoder produces a sequence of contextual representations:

$$
H = \operatorname{Encoder}(x_1, x_2, \ldots, x_S),
$$

with shape

$$
H \in \mathbb{R}^{B \times S \times d_{\text{model}}}.
$$

This $H$ is sometimes called the encoder memory or source memory.

It is not usually one vector representing the entire sentence. It is a collection of vectors:

$$
H = [h_1, h_2, \ldots, h_S].
$$

Each $h_i$ represents source position $i$, but it is contextualized using information from the whole source sequence. The decoder will attend to these vectors when deciding what to generate.

This is the first important correction to the phrase “the encoder creates a latent representation.” It does create latent representations, but normally one contextual representation per source position rather than one single compressed vector.

## Part 2: The decoder

The decoder has a stack of blocks, usually with the same number of layers as the encoder. Each decoder block contains **three** sublayers:

1. masked decoder self-attention;
2. encoder–decoder cross-attention;
3. position-wise feed-forward network.

Residual connections and layer normalization are used around these sublayers.

The decoder's job is different from the encoder's job:

- the encoder builds representations of the known source sequence;
- the decoder builds representations of the target prefix and uses the encoder output to predict the next target token.

### Step 1: Decoder input embeddings

During training, the decoder receives the shifted target sequence. For the example:

$$
[\text{<BOS>},\ \text{J'aime},\ \text{les},\ \text{chats}].
$$

The target tokens are embedded and combined with positional information. If the target length is $T$, the initial decoder representation has shape

$$
Y^{(0)} \in \mathbb{R}^{B \times T \times d_{\text{model}}}.
$$

### Step 2: Masked decoder self-attention

The decoder first performs self-attention over the target sequence. However, it uses a **causal mask** or **look-ahead mask**.

At position $t$, the decoder may see:

- the beginning-of-sequence token;
- target tokens at positions $1$ through $t-1$;
- its own current input representation, depending on the implementation's indexing convention.

It must not see target tokens at positions greater than $t$.

For a target sequence of length four, the allowed attention pattern is conceptually:

$$
\begin{bmatrix}
\checkmark & \times & \times & \times \\
\checkmark & \checkmark & \times & \times \\
\checkmark & \checkmark & \checkmark & \times \\
\checkmark & \checkmark & \checkmark & \checkmark
\end{bmatrix}.
$$

The first row can use only the first target input. The second row can use the first two target inputs, and so on.

In an implementation, forbidden attention logits are replaced by a very negative value before softmax, so their attention weights become approximately zero.

The purpose is to make training match the information restriction at generation time. If the decoder could see the correct future word during training, it could cheat instead of learning to predict it.

### Step 3: Encoder–decoder cross-attention

After masked self-attention, the decoder uses the encoder output. This is often called cross-attention or encoder–decoder attention.

The decoder representation produces the queries:

$$
Q = YW^Q.
$$

The encoder output produces the keys and values:

$$
K = HW^K, \qquad V = HW^V.
$$

Therefore:

- queries come from the current decoder target representation;
- keys come from the source representation;
- values come from the source representation.

The cross-attention operation is

$$
\operatorname{CrossAttention}(Y,H)
= \operatorname{softmax}\left(\frac{(YW^Q)(HW^K)^\top}{\sqrt{d_k}}\right)(HW^V).
$$

If the target length is $T$ and source length is $S$, the cross-attention score matrix has shape

$$
QK^\top \in \mathbb{R}^{T \times S}.
$$

This means every target position can decide which source positions are relevant. When generating a translation of “cats,” the decoder may attend strongly to the source token “cats.” When generating a word whose choice depends on the whole sentence, it can attend to several source positions.

Cross-attention is the bridge between the two stacks. The encoder does not directly generate target words. It provides source representations that the decoder queries while generating.

### Step 4: Decoder feed-forward network

After cross-attention, each target position passes through the same position-wise feed-forward network:

$$
\operatorname{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2.
$$

At this point, each decoder position contains information from:

- the target prefix through masked self-attention;
- the source sentence through cross-attention;
- nonlinear feature transformations through the feed-forward network.

After the final decoder block, the decoder produces contextual target representations:

$$
D \in \mathbb{R}^{B \times T \times d_{\text{model}}}.
$$

## Part 3: The final vocabulary projection

The decoder output is not yet a probability distribution. A learned linear projection maps each decoder hidden state from dimension $d_{\text{model}}$ to the vocabulary size $V$:

$$
Z = DW_{\text{vocab}} + b_{\text{vocab}},
$$

where

$$
W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times V}.
$$

The resulting logits have shape

$$
Z \in \mathbb{R}^{B \times T \times V}.
$$

For one target position, the model has a vector of $V$ scores, one for every vocabulary token. Applying softmax gives a probability distribution:

$$
p(y_t \mid y_{<t}, x)
= \operatorname{softmax}(z_t).
$$

The model therefore predicts a probability distribution over the next target token for every decoder position in the training sequence.

The vocabulary projection is often called the output projection or language-model head. In some implementations, its weights are tied to the target embedding matrix, but weight tying is an implementation choice rather than the central encoder–decoder idea.

## Training: parallel computation with teacher forcing

The phrase “the decoder takes one token at a time” is correct for inference but not for ordinary training.

During training, the complete target sentence is known. The model can receive the entire shifted target sequence at once:

$$
[\text{<BOS>},\ \text{J'aime},\ \text{les},\ \text{chats}].
$$

The desired outputs are also known:

$$
[\text{J'aime},\ \text{les},\ \text{chats},\ \text{<EOS>}].
$$

The decoder processes all four target positions in one forward pass. The causal mask ensures that each position can use only the appropriate prefix.

The predictions are aligned as follows:

| Decoder input available | Target to predict |
| --- | --- |
| `<BOS>` | `J'aime` |
| `<BOS> J'aime` | `les` |
| `<BOS> J'aime les` | `chats` |
| `<BOS> J'aime les chats` | `<EOS>` |

This is called **teacher forcing**: during training, the decoder receives the correct previous target tokens rather than its own earlier predictions.

The training loss is usually the sum or mean of cross-entropy losses over target positions:

$$
\mathcal{L}
= -\sum_{t=1}^{T}
\log p(y_t \mid y_{<t}, x).
$$

Because the causal mask hides future target tokens, all positions can be processed in parallel without allowing information leakage.

This is one of the main efficiency advantages of the Transformer over an RNN decoder during training. The target positions are causally constrained mathematically, but the matrix operations for all positions can still run in parallel on hardware.

## Inference: autoregressive generation one token at a time

At inference time, the correct target sentence is unknown. The decoder must generate it.

Generation starts with a beginning-of-sequence token:

$$
Y_0 = [\text{<BOS>}].
$$

The model predicts a distribution for the next token. Suppose it selects `J'aime`:

$$
Y_1 = [\text{<BOS>},\ \text{J'aime}].
$$

The model is called again, or incrementally updated, to predict the next token. Suppose it selects `les`:

$$
Y_2 = [\text{<BOS>},\ \text{J'aime},\ \text{les}].
$$

This continues until the model produces `<EOS>` or reaches a maximum length:

$$
\begin{aligned}
[\text{<BOS>}] &\rightarrow \text{J'aime}, \\
[\text{<BOS>},\ \text{J'aime}] &\rightarrow \text{les}, \\
[\text{<BOS>},\ \text{J'aime},\ \text{les}] &\rightarrow \text{chats}, \\
[\text{<BOS>},\ \text{J'aime},\ \text{les},\ \text{chats}] &\rightarrow \text{<EOS>}.
\end{aligned}
$$

At each step, the model uses:

1. the complete source representation from the encoder;
2. the target tokens generated so far;
3. masked self-attention over that generated prefix;
4. cross-attention from the current target representation to the encoder outputs;
5. the vocabulary projection and softmax.

Inference is sequential because the next input depends on the token generated at the previous step. This creates an unavoidable autoregressive dependency for this decoder design.

### Caching during inference

Implementations usually cache the key and value projections from previous decoder positions. This avoids recomputing the entire decoder self-attention history at every generation step.

Caching improves efficiency, but it does not make the semantic generation process fully parallel. Token $t+1$ still depends on the generated result at token $t$.

### Greedy decoding, beam search, and sampling

The probability distribution for the next token can be converted into a sequence in different ways:

- **Greedy decoding:** choose the highest-probability token at every step.
- **Beam search:** keep several high-probability partial sequences and expand them.
- **Sampling:** sample from the probability distribution, possibly after applying temperature or filtering.

These decoding strategies occur after the Transformer produces next-token probabilities. They are not separate encoder or decoder blocks.

## Training versus inference summary

| Component | Training | Inference |
| --- | --- | --- |
| Encoder source tokens | All processed in parallel | All processed in parallel because the source is known |
| Decoder target tokens | Entire shifted target sequence supplied at once | Only previously generated tokens are available |
| Decoder self-attention | Causal mask prevents future-token access | Naturally sees only the generated prefix, with causal masking still used in the architecture |
| Target-position computation | All positions computed in parallel | Positions generated sequentially |
| Previous target tokens | Correct tokens supplied by teacher forcing | Model's own previous predictions supplied |
| Vocabulary projection | Produces logits for every target position | Usually use the final/current position's logits for the next token |

This distinction is essential:

> **The decoder is parallelizable during training but autoregressive during inference.**

## What exactly is passed from encoder to decoder?

The encoder passes the complete sequence of contextual source vectors:

$$
H = [h_1, h_2, \ldots, h_S].
$$

It does not pass only the final token or a single compressed sentence vector in the original Transformer architecture.

At each decoder layer, the current target representations query this same encoder memory through cross-attention. Different target positions can attend to different source positions and with different weights.

For example:

- the decoder position generating a noun may attend strongly to the source noun;
- the decoder position generating an adjective may attend to the relevant source adjective;
- a position requiring broader context may attend to several source tokens.

The encoder outputs are therefore a reusable memory that the decoder consults at every target position and every decoder layer.

## How this relates to the ViT paper

The Vision Transformer uses the **encoder side** of the Transformer idea, not the original translation-style encoder–decoder generation process.

For image classification, ViT:

1. converts image patches into tokens;
2. adds positional embeddings and a class token;
3. runs Transformer encoder blocks;
4. reads the final class-token representation;
5. applies a classification head.

There is no autoregressive decoder because image classification does not require generating a target sequence one token at a time. The class token acts as a learned image-level readout instead of a decoder generating vocabulary probabilities.

The original Transformer and ViT therefore share the encoder-block ingredients—self-attention, feed-forward layers, residual connections, and normalization—but they solve different problems:

- original Transformer: encode a source sequence and autoregressively generate a target sequence;
- ViT: encode image-patch tokens and classify the image from an encoded representation.

## Final corrected mental model

Your proposed picture is close if refined as follows:

> The original Transformer has an encoder stack and a decoder stack. The encoder receives the complete source sequence, processes all source positions in parallel through repeated self-attention and feed-forward blocks, and outputs one contextual vector per source position. The decoder receives the shifted target sequence during training, applies masked self-attention, cross-attends to the encoder's sequence of outputs, applies a feed-forward network, and produces one hidden vector per target position. A shared linear vocabulary projection converts each decoder hidden vector into vocabulary logits and softmax probabilities. During training, all target positions are computed in parallel using a causal mask and teacher forcing. During inference, the target sequence is unknown, so the decoder generates one token at a time, feeding its previous predictions back into the next step.

Primary source: *Attention Is All You Need*, Sections 3.1–3.4. The ViT connection is an interpretation based on the encoder structure described in [ViT notes](notes.md) and the original Transformer source.

## Interaction 4 — Original Transformer encoder–decoder data flow
**Q:** Does the original Transformer encode the complete source sequence with an encoder stack, pass a latent representation to a decoder stack, and generate vocabulary probabilities one token at a time?
**A:** Corrected and expanded the mental model using *Attention Is All You Need*: the encoder outputs one contextual representation per source position, not one single latent vector; each decoder block contains masked self-attention, encoder–decoder cross-attention, and a feed-forward network; the vocabulary projection maps decoder states to logits. Clarified that the decoder is parallel during teacher-forced training with a causal mask but autoregressive one-token-at-a-time during inference. Connected this encoder-only distinction to ViT. Added the full explanation to the active ViT notes; no prototype or code changes were made.
