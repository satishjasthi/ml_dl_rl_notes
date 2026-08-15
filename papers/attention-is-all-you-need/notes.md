# Paper: Attention Is All You Need

- Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
- Published: arXiv 2017; NeurIPS 2017
- Primary link: https://arxiv.org/abs/1706.03762
- PDF link: https://arxiv.org/pdf/1706.03762
- Code/data links: Not recorded; no link was verified from the supplied arXiv source.
- Topics: attention, Transformer, sequence transduction, neural machine translation, encoder-decoder architectures
- Date started: 2026-08-14
- Study goals:
  - Understand the intuition — in progress
  - Implement a prototype — not started
  - Prepare for an interview/presentation — not started
- Background: Deep learning familiarity; beginner with attention
- Current priority: Understand the intuition first, then implement a prototype and prepare interview explanations
- Understanding status: In progress
- Source provenance: Metadata and high-level claims are based on the supplied arXiv source, including the abstract and Sections 1–3. Interpretive explanations are labeled as intuition or interpretation.

## Paper at a glance

The paper introduces the **Transformer**, an encoder-decoder architecture for sequence transduction that uses attention mechanisms instead of recurrence and convolution. The original experiments focus on machine translation: English-to-German and English-to-French. The authors argue that removing sequential recurrence makes training substantially more parallelizable while preserving or improving translation quality.

The key idea is to let each token build a context-aware representation by looking directly at other relevant tokens. In a sentence, a word does not have to pass information step by step through all the words between it and its useful context; attention creates direct, content-dependent connections.

## Intuition first

### The problem before the Transformer

Earlier sequence-to-sequence systems commonly used recurrent neural networks. An RNN reads tokens in order, updating a hidden state at each step. This gives a natural notion of sequence order, but it also creates a bottleneck: token $t$ cannot be processed until token $t-1$ has been processed. Long-range information must also travel through many recurrent steps.

The Transformer asks: **Can we process all token positions together and still let each token use the information it needs from the rest of the sentence?** Its answer is self-attention.

### What attention means in plain language

Suppose the sentence is “The animal did not cross the street because it was tired.” To interpret “it,” a model should consider “animal,” not just the immediately preceding word “street.” Attention gives the representation of “it” a way to assign more weight to useful words and less weight to irrelevant ones.

A useful mental model is a room full of tokens. Each token asks, “Which other tokens should I listen to, and by how much?” The answer depends on the token content and is recomputed for each input. This is different from a fixed window or a fixed set of neighbors.

### Query, key, and value analogy

Attention uses three learned views of each token:

- A **query** describes what this token is looking for.
- A **key** describes what each candidate token offers for matching.
- A **value** contains the information that will actually be gathered.

A query is compared with all keys. The resulting scores are normalized into weights, and the output is a weighted mixture of the values. In compact form:

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

Here, $Q$, $K$, and $V$ are matrices of queries, keys, and values; $d_k$ is the key-vector dimension; and the softmax turns compatibility scores into weights. The division by $\sqrt{d_k}$ prevents large dot products from making the softmax too sharp too early (paper Section 3.2.1).

### Self-attention

In **self-attention**, the queries, keys, and values all come from the same sequence. Therefore, every token can directly gather information from every other token in that sequence. The output at each position is still position-specific because each position has its own query and therefore its own attention weights.

This does not mean every token receives the same sentence representation. It means every token has access to the same pool of input positions, while learning a different, content-dependent mixture.

### Why multiple heads?

One attention operation may learn one type of relationship. **Multi-head attention** runs several attention operations in parallel, with different learned projections. One head may focus on a syntactic relationship, another on a distant reference, and another on local word compatibility. This is an intuition, not a guarantee that each head has one clean human-interpretable role.

The heads are concatenated and projected back into the model dimension (paper Section 3.2.2).

### Why positional encoding is necessary

Attention by itself does not know whether a token came first, second, or last. If the same token representations were presented in a different order, the basic attention operation would not inherently distinguish the permutations. The Transformer therefore adds positional encodings to token embeddings. In the original paper, these are sinusoidal functions of position and dimension (paper Section 3.5).

### The architecture at a glance

The original Transformer has:

1. An **encoder stack** that reads the source sequence.
2. A **decoder stack** that generates the target sequence.
3. Encoder self-attention, allowing source tokens to use one another.
4. Decoder masked self-attention, preventing a target position from seeing future target tokens during training.
5. Encoder-decoder attention, allowing each decoder position to retrieve information from the encoded source sentence.
6. Position-wise feed-forward networks, residual connections, and layer normalization (paper Section 3.1).

The decoder is autoregressive at generation time: it predicts one target token, then uses that output as part of the next step. Training can nevertheless process all target positions in parallel because a causal mask hides future target tokens.

## First interview-ready explanation

> The Transformer is an encoder-decoder sequence model based entirely on attention. Self-attention lets every token directly weigh information from all other tokens, so the model can capture long-range dependencies without passing information sequentially through an RNN. Because token positions can be processed in parallel, training is faster and more scalable. Since attention alone has no notion of order, positional encodings are added, and the decoder uses masking to preserve autoregressive generation.

## What the paper claims versus what this first lesson interprets

- **Paper claim:** The Transformer is based solely on attention mechanisms and dispenses with recurrence and convolution.
- **Paper evidence:** On WMT 2014 translation tasks, the reported Transformer models achieve strong BLEU scores and require less training time than the compared systems (paper abstract and Section 4).
- **Interpretation:** Attention is best understood initially as content-dependent information routing: each token selects and combines useful information from other positions.
- **Important qualification:** Attention weights are useful diagnostic signals, but they should not automatically be treated as complete explanations of a model's reasoning.

## Current understanding

The initial goal is to understand attention before implementing it. The current mental model is: queries ask, keys determine relevance, values provide content, softmax converts relevance into mixing weights, and self-attention performs this operation among tokens in one sequence. The next useful step is a small numerical example of one attention head, followed by a minimal prototype.

## Interaction log

## Interaction 1 — Paper setup and intuition roadmap
**Q:** Set up the Attention Is All You Need study workspace with goals to understand intuition, implement a prototype, and prepare for an interview.
**A:** Confirmed the active paper and notes path. Created the paper workspace and recorded the source, metadata, independent goal statuses, and beginner-friendly intuition lesson. The first explanation frames attention as content-dependent information routing, introduces queries, keys, values, self-attention, multi-head attention, positional encoding, and decoder masking, and distinguishes paper claims from interpretation. No prototype was created because implementation has not yet been explicitly requested.


## Attention from Scratch: A Bottom-Up Walkthrough

This handout builds one attention head from the smallest useful example.

We will use the sentence:

> **The cat sat.**

We will focus on the token **sat** and ask:

> Which other tokens should `sat` use to understand its context?

## The complete attention pipeline

1. Convert tokens into vectors.
2. Create queries, keys, and values.
3. Compare one query with every key.
4. Scale the scores by `sqrt(d_k)`.
5. Apply softmax to get attention weights.
6. Use the weights to combine the values.
7. Produce a context-aware representation.

## 1. Tokens become vectors

After tokenization, the sentence is represented as:

```text
[The, cat, sat]
```

Each token is converted into a vector:

$$
x_{\text{The}}, \qquad x_{\text{cat}}, \qquad x_{\text{sat}}
$$

In this handout, keep two different dimensions separate:

- **Sequence length, $L$:** how many token positions are in the current input. For `The cat sat`, $L=3$.
- **Key dimension, $d_k$:** how many numbers are in one key vector for one attention head. In this toy example, we choose $d_k=2$, so every key has two components.

These are independent. A sequence can contain 3, 100, or 4,096 tokens while the same model head still uses the same $d_k$. Conversely, changing the model architecture can change $d_k$ without changing the number of input tokens.

For one attention head, the shapes are:

$$
Q \in \mathbb{R}^{L \times d_k},
\qquad
K \in \mathbb{R}^{L \times d_k},
\qquad
V \in \mathbb{R}^{L \times d_v}
$$

The score matrix has shape:

$$
QK^\top:
(L \times d_k)(d_k \times L)=L \times L
$$

So:

- $L$ determines **how many rows and columns** are in the attention-score matrix.
- $d_k$ determines the **length of the vectors being dotted together** and therefore the scale factor $\sqrt{d_k}$.

### How does a real model choose $d_k$?

For ordinary multi-head attention, the model configuration usually provides:

- $d_{\text{model}}$: the hidden/embedding width
- $H$: the number of query attention heads

When the hidden width is evenly divided across heads:

$$
 d_k = \text{head\_dim} = \frac{d_{\text{model}}}{H}
$$

For example:

| Model configuration | Calculation | Per-head $d_k$ |
|---|---:|---:|
| $d_{\text{model}}=768$, $H=12$ | $768/12$ | $64$ |
| $d_{\text{model}}=4096$, $H=32$ | $4096/32$ | $128$ |

In code, prefer an explicit `head_dim` or `hidden_size / num_attention_heads` from the model configuration. Do **not** calculate $d_k$ from the number of input tokens.

Grouped-query attention may use fewer key/value heads than query heads, but the key vector width is still typically the configured `head_dim`. Some newer architectures use special attention variants, so their model configuration and implementation are the final authority.

For the toy sentence, the facts are therefore:

```text
number of tokens (sequence length L) = 3
key dimension per head (d_k)        = 2  ← chosen for the example
```

## 2. Every token gets three views

For every token representation $x_i$, the model creates three vectors:

$$
q_i = x_i W^Q
$$

$$
k_i = x_i W^K
$$

$$
v_i = x_i W^V
$$

These vectors have different jobs:

- **Query:** What is this token looking for?
- **Key:** How can this token be found by other tokens?
- **Value:** What information should this token provide if selected?

### Library analogy

- The **query** is a search request.
- The **key** is the label or index of a book.
- The **value** is the actual content inside the book.

Queries and keys decide what matches. Values provide the information retrieved after matching.

The matrices $W^Q$, $W^K$, and $W^V$ are learned during training. They create different roles from the same token representation.

## 3. Focus on the query for `sat`

Suppose the learned query for `sat` is:

$$
q_{\text{sat}} = [2, 1]
$$

Suppose the keys are:

$$
k_{\text{The}} = [0, 1]
$$

$$
k_{\text{cat}} = [2, 1]
$$

$$
k_{\text{sat}} = [1, 0]
$$

These numbers are illustrative. Real query and key vectors are learned floating-point values.

The query for `sat` is compared with every key using a dot product.

## 4. Calculate query-key compatibility scores

For `The`:

$$
q_{\text{sat}} k_{\text{The}}^\top
= [2,1][0,1]^\top
= 2(0)+1(1)
= 1
$$

For `cat`:

$$
q_{\text{sat}} k_{\text{cat}}^\top
= [2,1][2,1]^\top
= 2(2)+1(1)
= 5
$$

For `sat` itself:

$$
q_{\text{sat}} k_{\text{sat}}^\top
= [2,1][1,0]^\top
= 2(1)+1(0)
= 2
$$

The raw scores are therefore:

$$
[1, 5, 2]
$$

The largest score is for `cat`. In this toy example, the query for `sat` considers `cat` the most relevant token.

The model does not receive a hard-coded grammar rule saying that `cat` is the subject of `sat`. During training, it learns projections that make useful relationships produce larger scores.

## 5. Scale by `sqrt(d_k)`

Here, each query and key has dimension $d_k=2$ because the toy vectors contain two numbers each. This is **not** because the sentence has three tokens; the sentence length is $L=3$, while $d_k$ is the width of each per-head query/key vector.

Therefore:

$$
\sqrt{d_k} = \sqrt{2}
$$

The scaled scores are:

$$
\frac{[1,5,2]}{\sqrt{2}}
\approx
[0.707, 3.536, 1.414]
$$

Why scale? Dot products tend to become larger as the vector dimension $d_k$ grows. Very large scores make softmax extremely sharp and can make learning unstable. Dividing by $\sqrt{d_k}$ keeps the scores in a more manageable range.

## 6. Softmax creates attention weights

Apply softmax to the scaled scores:

$$
\operatorname{softmax}([0.707,3.536,1.414])
\approx
[0.05,0.85,0.10]
$$

The weights are:

| Token | Attention weight |
|---|---:|
| `The` | $0.05$ |
| `cat` | $0.85$ |
| `sat` | $0.10$ |

The weights sum to one:

$$
0.05+0.85+0.10=1
$$

So, while processing `sat`, the model attends approximately:

- 5% to `The`
- 85% to `cat`
- 10% to `sat` itself

These are not permanent importance scores. A different query, such as the query for `cat`, can produce a different distribution.

## 7. Use the weights to retrieve values

Suppose the value vectors are:

$$
v_{\text{The}} = [1,0]
$$

$$
v_{\text{cat}} = [0,3]
$$

$$
v_{\text{sat}} = [2,1]
$$

The output for `sat` is the weighted sum of these values:

$$
o_{\text{sat}}
= 0.05v_{\text{The}}
+ 0.85v_{\text{cat}}
+ 0.10v_{\text{sat}}
$$

Substituting the values:

$$
o_{\text{sat}}
= 0.05[1,0]
+ 0.85[0,3]
+ 0.10[2,1]
$$

Therefore:

$$
o_{\text{sat}} = [0.25, 2.65]
$$

This is now a **context-enriched representation** of `sat`. It contains mostly information retrieved from `cat` because `cat` received the largest attention weight.

The key distinction is:

- **Keys** help decide where to look.
- **Values** provide the information retrieved from those locations.

## 8. The complete matrix calculation

The previous calculation focused on one query. In practice, the Transformer calculates attention for all tokens at once.

Stack the queries, keys, and values into matrices:

$$
Q =
\begin{bmatrix}
q_{\text{The}}\\
q_{\text{cat}}\\
q_{\text{sat}}
\end{bmatrix},
\qquad
K =
\begin{bmatrix}
k_{\text{The}}\\
k_{\text{cat}}\\
k_{\text{sat}}
\end{bmatrix},
\qquad
V =
\begin{bmatrix}
v_{\text{The}}\\
v_{\text{cat}}\\
v_{\text{sat}}
\end{bmatrix}
$$

The complete scaled dot-product attention equation is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

The steps are:

1. $QK^\top$ calculates every query-key compatibility score.
2. Division by $\sqrt{d_k}$ stabilizes the score magnitudes.
3. Row-wise softmax converts each query's scores into weights.
4. Multiplication by $V$ creates weighted combinations of value vectors.

For one head, if there are $L$ tokens and the query/key vectors have width $d_k$:

$$
Q \in \mathbb{R}^{L \times d_k},
\qquad
K \in \mathbb{R}^{L \times d_k},
\qquad
V \in \mathbb{R}^{L \times d_v}
$$

Therefore:

$$
QK^\top:
(L \times d_k)(d_k \times L)
= L \times L
$$

The resulting matrix has one row per querying token and one column per candidate token. The `sat` row tells us how much `sat` should consider `The`, `cat`, and `sat`.

Notice what changes each dimension:

- Increasing the number of input tokens changes $L$ and makes the attention matrix larger: $L \times L$.
- Changing the model's per-head width changes $d_k$ and the amount of computation inside each dot product.
- The scaling denominator is $\sqrt{d_k}$, **not** $\sqrt{L}$.

For a batch of multi-head attention, a common layout is:

$$
Q \in \mathbb{R}^{B \times H \times L \times d_k},
\qquad
K \in \mathbb{R}^{B \times H \times L \times d_k},
\qquad
V \in \mathbb{R}^{B \times H \times L \times d_v}
$$

where $B$ is batch size, $H$ is the number of heads, and $L$ is the token count per sequence. The attention scores then have shape:

$$
B \times H \times L \times L
$$

Usually $d_k=d_v=\text{head\_dim}$, but the value width can be different in principle.

## Why are Q, K, and V separate?

Using separate projections gives the model flexibility:

- The **query** can encode what a token needs.
- The **key** can encode how a token should be matched.
- The **value** can encode what information should be passed onward.

If the same vector had to perform all three jobs, matching and information retrieval would be less flexible. Separate learned projections let the model use one representation as a search address and another as the payload.

## The whole intuition in one paragraph

Every token searches the sequence for useful information. Its query describes what it needs. Every token's key describes how it can be found. Dot products measure compatibility between the query and each key. Softmax turns those scores into normalized weights. The weights determine how much information to retrieve from each value. The weighted sum becomes a new representation that understands the token in context.

This is the core of one attention head. Multi-head attention repeats this process several times in parallel, allowing different heads to learn different types of relationships.


## Interaction 2 — Attention diagnostic quiz
**Q:** Test understanding of attention concepts and Transformer fundamentals from basic ideas through application-level and complex scenario questions.
**A:** Prepared an interactive, source-aware diagnostic in progressive rounds, beginning with foundations and reserving tensor-shape, implementation, masking, positional-encoding, and scenario questions for later rounds. No paper facts or workspace metadata were changed.


## Interaction 3 — Diagnostic questions workspace
**Q:** Why were the attention diagnostic questions not added to `questions.md` in the paper workspace?
**A:** Clarified that `questions.md` had been treated too narrowly as an unresolved-concepts backlog. Added the active Round 1 foundations diagnostic there, preserving the existing questions, and will use the file to track future quiz rounds and response status.


## Interaction 4 — Complete diagnostic bank and answer key
**Q:** Update all attention quiz rounds in `questions.md` and generate answers based on the paper sections and questions.
**A:** Expanded `questions.md` with six progressive rounds covering foundations, core mechanics, architecture and training, implementation and debugging, application trade-offs, and complex scenarios. Created `answers.md` with answers to the five existing unresolved questions and every question in all six rounds, while distinguishing paper claims from implementation guidance and interpretation. Read back both files and corrected/validated their Markdown LaTeX formatting.
