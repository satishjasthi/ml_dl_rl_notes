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
= \operatorname{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right)V
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

## Attention deep dive — intuition, math, and a small example

### 1. The central idea

Attention is a learned information-retrieval operation. For each token, the model asks:

1. **What am I looking for?** — the query.
2. **Which other tokens match that need?** — compare the query with keys.
3. **What information should I retrieve from those tokens?** — combine their values.

The important point is that attention weights are **query-dependent**. There is not one permanent importance score for each word. The same word can be highly relevant to one token and irrelevant to another.

### 2. From token representations to queries, keys, and values

Let a sequence contain $n$ tokens, with each token represented using a $d_{\text{model}}$-dimensional vector. Stack the token vectors into:

$$
X \in \mathbb{R}^{n \times d_{\text{model}}}
$$

The model learns three projection matrices:

$$
W^Q \in \mathbb{R}^{d_{\text{model}} \times d_k},
\qquad
W^K \in \mathbb{R}^{d_{\text{model}} \times d_k},
\qquad
W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

It projects the same input sequence into three roles:

$$
Q = XW^Q,
\qquad
K = XW^K,
\qquad
V = XW^V
$$

Therefore:

$$
Q,K \in \mathbb{R}^{n \times d_k},
\qquad
V \in \mathbb{R}^{n \times d_v}
$$

The token itself does not permanently contain a query, key, or value. These are learned, context-processing views created by the projection matrices. This is the scaled dot-product attention construction in paper Section 3.2.1.

### 3. Computing relevance scores

For token $i$ and candidate token $j$, the raw compatibility score is the dot product:

$$
s_{ij} = q_i k_j^{\mathsf{T}}
$$

A large positive score means the query and key point in similar directions in the learned feature space. A small or negative score means they are less compatible.

Computing all pairwise scores at once gives:

$$
S = QK^{\mathsf{T}}
$$

The shapes make the operation clear:

$$
(n \times d_k)(d_k \times n) = n \times n
$$

There is one score for every query position and every candidate key position. Row $i$ describes how token $i$ relates to all tokens.

### 4. Why divide by $\sqrt{d_k}$?

The paper scales the scores before applying softmax:

$$
Z = \frac{QK^{\mathsf{T}}}{\sqrt{d_k}}
$$

If the components of $q_i$ and $k_j$ are independent, zero-mean, and have variance approximately $1$, the dot product is a sum of $d_k$ products. Its variance grows approximately with $d_k$, so its typical magnitude grows with $\sqrt{d_k}$.

Without scaling, larger key dimensions can produce large logits. Softmax then becomes nearly one-hot: one candidate gets almost all the weight, the others get almost none, and gradients become less useful. Dividing by $\sqrt{d_k}$ keeps the scores in a more stable range. This is the paper's motivation in Section 3.2.1.

### 5. Turning scores into attention weights

Softmax is applied independently across each row, meaning each query distributes a total probability mass of $1$ over candidate tokens:

$$
\alpha_{ij}
= \frac{\exp\left(s_{ij}/\sqrt{d_k}\right)}
{\sum_{r=1}^{n}\exp\left(s_{ir}/\sqrt{d_k}\right)}
$$

For every query position $i$:

$$
\sum_{j=1}^{n}\alpha_{ij}=1
\qquad\text{and}\qquad
\alpha_{ij}\ge 0
$$

Stacking all weights into a matrix gives:

$$
A = \operatorname{softmax}_{\text{row}}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right)
$$

The weights answer **how much each query listens to each candidate**. They are not the final output yet.

### 6. Retrieving and mixing values

The final attention output is:

$$
O = AV
$$

The shape is:

$$
(n \times n)(n \times d_v)=n \times d_v
$$

For a single query position $i$, this is:

$$
o_i = \sum_{j=1}^{n}\alpha_{ij}v_j
$$

So each output is a weighted average of the value vectors. The keys decide relevance; the values supply the retrieved content.

Putting all steps together gives the paper's main equation:

$$
\operatorname{Attention}(Q,K,V)
= \operatorname{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right)V
$$

### 7. Numerical example with two tokens

To focus on the mechanics, assume the projections have already happened. Consider two query positions and two candidate values:

$$
Q =
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix},
\qquad
K =
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix},
\qquad
V =
\begin{bmatrix}
10 & 0\\
0 & 20
\end{bmatrix}
$$

Here $d_k=2$, so $\sqrt{d_k}=\sqrt{2}\approx 1.414$.

#### Step 1: raw query-key scores

$$
QK^{\mathsf{T}}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}^{\mathsf{T}}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
$$

The first query matches key 1 with score $1$ and key 2 with score $0$. The second query has the opposite preference.

#### Step 2: scale the scores

$$
Z = \frac{QK^{\mathsf{T}}}{\sqrt{2}}
\approx
\begin{bmatrix}
0.707 & 0\\
0 & 0.707
\end{bmatrix}
$$

#### Step 3: apply row-wise softmax

For the first row:

$$
\operatorname{softmax}([0.707,0])
= \left[
\frac{\exp(0.707)}{\exp(0.707)+\exp(0)},
\frac{\exp(0)}{\exp(0.707)+\exp(0)}
\right]
\approx [0.670,0.330]
$$

For the second row, the weights are reversed:

$$
A \approx
\begin{bmatrix}
0.670 & 0.330\\
0.330 & 0.670
\end{bmatrix}
$$

#### Step 4: mix the value vectors

$$
O=AV
\approx
\begin{bmatrix}
0.670 & 0.330\\
0.330 & 0.670
\end{bmatrix}
\begin{bmatrix}
10 & 0\\
0 & 20
\end{bmatrix}
=
\begin{bmatrix}
6.70 & 6.60\\
3.30 & 13.40
\end{bmatrix}
$$

Interpretation:

- Query 1 mostly listens to value 1, so its output is closer to $[10,0]$.
- Query 2 mostly listens to value 2, so its output is closer to $[0,20]$.
- Neither output simply copies one value because softmax gives both candidates some weight.

This example is intentionally small and uses already-projected vectors. In a real Transformer, $Q$, $K$, and $V$ are learned projections of token embeddings and the model learns the projections during training.

### 8. Self-attention, cross-attention, and masking

In encoder self-attention, all three inputs come from the same sequence:

$$
Q=XW^Q,\qquad K=XW^K,\qquad V=XW^V
$$

In decoder self-attention, the same is true, but a causal mask prevents position $i$ from using future positions $j>i$. A mask can be represented as:

$$
M_{ij}=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

The masked operation is:

$$
A=\operatorname{softmax}_{\text{row}}
\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}+M\right)
$$

Adding $-\infty$ makes the corresponding softmax probability zero. Thus, all target positions can be computed in parallel during training, while each position still behaves as if it only knew the preceding target tokens.

In encoder-decoder attention, queries come from the decoder and keys and values come from the encoder output $H$:

$$
Q=YW^Q,\qquad K=HW^K,\qquad V=HW^V
$$

This lets each generated target token retrieve relevant information from the encoded source sentence.

### 9. Multi-head attention

Instead of using one attention operation, the Transformer uses $h$ heads. Each head has its own projections:

$$
\operatorname{head}_r
=\operatorname{Attention}
\left(XW_r^Q,XW_r^K,XW_r^V\right)
$$

The heads are concatenated and projected:

$$
\operatorname{MultiHead}(X)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
$$

Different heads can learn different relationships, but the interpretation that one head always corresponds to one human concept is only a possible diagnostic pattern, not a guaranteed property.

### 10. What attention does and does not do

- Attention does **not** create information from nowhere; it routes and mixes value representations after learned projections.
- Attention weights are not fixed word importance scores; they depend on the query, input, layer, head, and masking.
- Attention alone does not encode order; the Transformer adds positional encodings as described in Section 3.5.
- Self-attention gives direct pairwise interaction between positions, but its score matrix has $n^2$ entries, so computation and memory grow quadratically with sequence length.
- The Transformer gains training parallelism because all query-key-value operations for a sequence can be computed as matrix operations, rather than waiting for recurrent time steps.

## Interaction 3 — Attention intuition, derivation, and numerical example
**Q:** Explain attention in detail with intuition, a small example, and the complete mathematics.
**A:** Added a source-aware attention deep dive covering query-key-value projections, score and weight computation, $\sqrt{d_k}$ scaling, value aggregation, tensor shapes, a two-token numerical example, causal masking, cross-attention, multi-head attention, and important qualifications. The active concepts remain unchecked because the user has not explicitly confirmed mastery.
