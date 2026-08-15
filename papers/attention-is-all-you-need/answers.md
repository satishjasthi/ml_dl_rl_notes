# Answers: Attention Is All You Need

This answer key corresponds to every question currently listed in `questions.md`. Attempt each round before checking the answers. Some answers distinguish the original paper’s claims from implementation guidance or interpretation.

**Primary source:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762), especially Sections 3.1–3.5. The tensor-layout and debugging advice is implementation guidance derived from the paper’s equations and shapes, not always a verbatim paper prescription.

## Existing unresolved questions

### 1. How does the scaled dot-product attention equation produce one output vector for each token?

For a sequence of length $L$, $QK^\top$ produces an $L \times L$ score matrix. Each row corresponds to one querying token and contains that token’s compatibility with every candidate key. Row-wise softmax turns each row into weights that sum to one. Multiplying the resulting weight matrix by $V$ produces an $L \times d_v$ matrix, so there is one output vector for each query position:

$$
A = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right),
\qquad
O = AV.
$$

The $i$-th output is a weighted sum of value vectors using the $i$-th attention row.

### 2. Why does dividing by $\sqrt{d_k}$ stabilize softmax attention?

As the key/query dimension grows, the dot product can have larger variance and magnitude. Large logits make softmax very peaked: one position can receive almost all the weight, producing small gradients for the other positions. Dividing by $\sqrt{d_k}$ keeps the logits in a more manageable range and makes optimization less likely to saturate early. This is the motivation given in Section 3.2.1.

### 3. How exactly does the decoder mask prevent information leakage during training?

Before softmax, logits for future target positions are replaced by a value behaving like $-\infty$. For target position $i$, only positions $1$ through $i$ remain visible in masked self-attention. Their weights are normalized among themselves, while future positions receive effectively zero weight. Because all target positions can still be computed in one matrix operation, training remains parallel, but each prediction has only past-and-current target context.

### 4. What are the tensor shapes of $Q$, $K$, $V$, scores, and outputs in a batched implementation?

A common multi-head layout is:

$$
Q,K \in \mathbb{R}^{B \times H \times L \times d_k},
\qquad
V \in \mathbb{R}^{B \times H \times L \times d_v}.
$$

The score and weight tensors have shape $B \times H \times L \times L$, and the per-head output has shape $B \times H \times L \times d_v$. After transposing and concatenating heads, the usual output is $B \times L \times H d_v$, followed by the output projection.

### 5. Which simplifications are acceptable in a small prototype, and which would change Transformer behavior?

A teaching prototype can use a tiny vocabulary, fixed-length toy inputs, one head, manually chosen vectors, and a single attention layer to demonstrate the equation. It can omit training, a full encoder-decoder stack, and production concerns. To still represent scaled attention, it should retain query-key scoring, division by $\sqrt{d_k}$, row-wise softmax, and weighted value aggregation. Omitting positional information, causal masking in a decoder, multi-head structure, residual/normalization blocks, or the feed-forward layers is acceptable only if explicitly labeled as a simplification; those omissions mean it is not a faithful reproduction of the full Transformer.

## Round 1 — Foundations

### 1. What problem does attention solve that is difficult for a plain RNN?

An RNN processes positions sequentially, so position $t$ waits for position $t-1$. Long-range information must also pass through many recurrent transitions. Self-attention lets every position access every other position through direct, content-dependent connections and allows all source positions to be processed together during training.

### 2. What are the roles of $Q$, $K$, and $V$?

A query describes what the current token is looking for. A key describes how a candidate token can be matched. A value contains the information retrieved after the candidate is selected. Queries and keys determine routing; values carry the payload. Separate learned projections allow matching features and transmitted information to serve different purposes.

### 3. Where do $Q$, $K$, and $V$ come from in self-attention?

They are learned linear projections of the same sequence representations:

$$
Q = XW^Q,
\qquad
K = XW^K,
\qquad
V = XW^V.
$$

Two tokens can produce different outputs because they have different queries, so they generate different attention-weight rows even though they attend to the same set of positions.

### 4. What happens to scaled scores $[2,1,0]$ under softmax?

The first token receives the largest weight because it has the largest score. The weights are positive and sum to one, approximately $[0.665, 0.245, 0.090]$. Multiplying those weights by $V$ forms a weighted average: the output is influenced most by the first value, but it can still contain information from all three values.

### 5. Why divide by $\sqrt{d_k}$?

The dot product generally grows in scale with the vector dimension. Without scaling, large $d_k$ can produce very large logits and an overly sharp softmax, which harms gradient flow. The denominator is determined by the key width $d_k$, not by the number of tokens.

### 6. Shapes for $L=5$, $d_k=64$, and $d_v=32$

For one head:

$$
Q \in \mathbb{R}^{5 \times 64},
\qquad
K \in \mathbb{R}^{5 \times 64},
\qquad
V \in \mathbb{R}^{5 \times 32}.
$$

Therefore,

$$
QK^\top \in \mathbb{R}^{5 \times 5},
\qquad
\operatorname{Attention}(Q,K,V) \in \mathbb{R}^{5 \times 32}.
$$

The sequence length controls the two dimensions of the score matrix; the key and value widths control vector dimensions.

### 7. Why is positional encoding needed?

Plain self-attention is content-based and does not inherently identify whether a token was first, second, or last. Without positional information, permuting the same token representations can leave the operation unable to represent the difference in order. Positional encodings inject position-dependent signals into the token representations. The paper uses sinusoidal encodings and also discusses learned positional embeddings.

### 8. What supplies $Q$, $K$, and $V$ in decoder encoder-decoder attention?

The decoder’s current representations supply the queries. The encoder’s output sequence supplies both keys and values. Thus each target position asks a source-conditioned question and retrieves information from the encoded source sentence. This is different from decoder self-attention, where all three come from the decoder-side sequence.

## Round 2 — Core mechanics and numerical reasoning

### 1. Numerical scores for the toy query

The raw scores are:

$$
[2,1] \cdot [0,1] = 1,
\qquad
[2,1] \cdot [2,1] = 5,
\qquad
[2,1] \cdot [1,0] = 2.
$$

So the raw score vector is $[1,5,2]$. With $d_k=2$:

$$
\frac{[1,5,2]}{\sqrt{2}}
\approx
[0.707,3.536,1.414].
$$

The second key has the largest compatibility score.

### 2. Why is softmax row-wise?

Each row represents one query position asking where to retrieve information. That query needs one normalized distribution over candidate keys. Normalizing the entire matrix would make unrelated queries compete with one another and would not produce an independent attention distribution for each output position.

### 3. How does the matrix equation produce one output per query?

Let $A=\operatorname{softmax}(QK^\top/\sqrt{d_k})$. If $Q$ has shape $L \times d_k$, $K$ has shape $L \times d_k$, and $V$ has shape $L \times d_v$, then $A$ has shape $L \times L$. The product $AV$ has shape $L \times d_v$. Row $i$ of $A$ supplies the weights for output vector $i$.

### 4. Batched multi-head shapes

The tensors are:

$$
Q,K \in \mathbb{R}^{2 \times 8 \times 16 \times 64},
\qquad
V \in \mathbb{R}^{2 \times 8 \times 16 \times 64}.
$$

The score and weight tensors have shape $2 \times 8 \times 16 \times 16$, and the per-head output has shape $2 \times 8 \times 16 \times 64$ before head concatenation.

### 5. Why may $d_v$ differ from $d_k$?

The key width controls the query-key dot product and therefore the scaling factor. The value width controls the dimensionality of the information retrieved. Matching and payload representation are separate design choices, although the original Transformer commonly uses equal per-head widths.

### 6. Multi-head computation and output projection

For head $i$:

$$
\operatorname{head}_i
=
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V).
$$

The heads are concatenated and projected:

$$
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_H)W^O.
$$

The output projection mixes information across head subspaces and returns the result to the model dimension needed by the residual stream.

### 7. Causal visibility for four decoder tokens

Using one-based positions, token 1 may attend to position 1; token 2 may attend to 1–2; token 3 may attend to 1–3; and token 4 may attend to 1–4. The visibility pattern is lower triangular. Therefore token 3 cannot attend to token 4, and its future logits are masked before softmax.

### 8. Cross-attention shapes

The target/decode sequence supplies $Q$ with shape $B \times H \times L_t \times d_k$. The encoded source supplies $K$ with shape $B \times H \times L_s \times d_k$ and $V$ with shape $B \times H \times L_s \times d_v$. The score tensor is:

$$
B \times H \times L_t \times L_s.
$$

Each target query can therefore select among source positions.

### 9. Identical values

If every value is the same vector $v$, then every output is $v$, because each attention row sums to one:

$$
\sum_j a_{ij}v = \left(\sum_j a_{ij}\right)v = v.
$$

The attention weights may differ, but they cannot change the weighted sum when all payload vectors are identical.

## Round 3 — Transformer architecture and training behavior

### 1. Information flow through encoder and decoder layers

An encoder layer applies self-attention over the source sequence, then a position-wise feed-forward network, with residual connections and layer normalization around the sublayers as described in Section 3.1. A decoder layer applies masked self-attention over prior target positions, encoder-decoder attention over the encoder outputs, and then a position-wise feed-forward network, again with residual and normalization blocks.

### 2. Three attention types

- **Encoder self-attention:** source queries, keys, and values all come from the source sequence; source positions can use one another.
- **Decoder masked self-attention:** queries, keys, and values come from the target side, but future target positions are hidden.
- **Encoder-decoder attention:** decoder states supply queries; encoder outputs supply keys and values; target positions retrieve source information.

### 3. Training parallelism versus inference

During training, the complete target sequence is available. A causal mask lets all target positions compute their predictions in parallel while preventing each position from seeing future target tokens. During autoregressive inference, the next token is not known until the previous token has been generated, so generation normally proceeds step by step.

### 4. What causal masking hides

For prediction at position $i$, the mask hides target positions after $i$. Teacher forcing supplies the correct previous target tokens, but it does not make future target tokens valid inputs to the current prediction. Without the mask, the network could inspect the answer it is supposed to predict, producing leakage and an unrealistically easy training task.

### 5. Residual connections and layer normalization

Residual connections provide a direct path for information and gradients around each sublayer, making deep stacks easier to optimize. Layer normalization stabilizes the activation scale within each token representation and supports more reliable optimization. They do not replace attention; they make the stacked transformations trainable and preserve useful information flow.

### 6. Position-wise feed-forward network

The same feed-forward network is applied independently to each position. It transforms features within a token representation but does not mix information across positions by itself. Cross-position interaction comes from attention; the feed-forward sublayer then performs nonlinear feature transformation at each position.

### 7. Positional encodings and the sinusoidal choice

Position encodings give the model information about order, which attention alone lacks. The original sinusoidal construction is:

$$
\operatorname{PE}(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right),
$$

$$
\operatorname{PE}(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right).
$$

These vectors are added to token embeddings. The paper also compares learned positional embeddings and reports similar results in its experiments; the important architectural requirement is that position information is supplied.

### 8. Parallelism and long-range dependencies

Recurrence imposes sequential dependence across positions. Self-attention computes all pairwise interactions for a layer at once, making training highly parallelizable. It also creates a short, direct computational path between distant positions instead of requiring information to pass through many recurrent steps. The trade-off is the quadratic number of pairwise interactions in sequence length.

### 9. Why multiple heads help

Each head has its own learned projections and can form a different attention distribution. Heads can therefore route different kinds of information in parallel—for example, local compatibility, syntactic relations, or long-distance dependencies as an interpretation. This is not a guarantee that each head has one clean human-readable role.

### 10. Which components serve which roles?

The encoder builds context-rich source representations. Decoder masked self-attention models the target-side prefix while preserving causality. Encoder-decoder attention aligns or retrieves source information for each target position. The decoder feed-forward layers transform the retrieved features, and the final output projection produces next-token probabilities.

## Round 4 — Implementation and debugging

### 1. Reshaping into heads

Starting with $[B,L,d_{\text{model}}]$ and $d_{\text{model}}=H d_k$, project to $[B,L,Hd_k]$. Reshape to $[B,L,H,d_k]$, then transpose to the convenient attention layout $[B,H,L,d_k]$. The same process is applied to keys and values, with $d_v$ replacing $d_k$ for values if they differ.

### 2. Incorrect score shape

For the common layout $Q,K \in [B,H,L,d_k]$, the last two dimensions must multiply as $[L,d_k][d_k,L]$, producing scores of shape $[B,H,L,L]$. A score shape $[B,H,d_k,L]$ suggests a missing or incorrect transpose of $K$, or an incorrect contraction dimension.

### 3. Mask placement

Apply the mask to the logits before softmax. Set future or invalid-key logits to a sufficiently negative value, conceptually $-\infty$. After softmax, those positions have approximately zero weight. Masking after softmax would leave the row improperly normalized unless the weights were renormalized carefully.

### 4. Stable softmax

For logits $z$, compute softmax using $z-\max(z)$ per row. Subtracting the same constant does not change the mathematical softmax distribution, but it prevents large positive exponentials from overflowing and improves numerical stability.

### 5. Scaling by $\sqrt{L}$ is wrong

The scaling compensates for the width of the vectors participating in each dot product, so the denominator is $\sqrt{d_k}$. Sequence length affects how many scores exist, not the expected scale of an individual query-key dot product. The bug may be hard to notice for fixed small inputs or when logits are not yet large, but it becomes harmful when sequence length changes or $L$ and $d_k$ differ substantially.

### 6. Why transpose, concatenate, and project heads?

Attention commonly produces $[B,H,L,d_v]$. Transpose to put sequence position before heads, giving $[B,L,H,d_v]$, then concatenate the head feature dimensions to obtain $[B,L,Hd_v]$. The output projection maps this combined representation back to the model dimension and mixes information across heads before residual addition.

### 7. Residual shape invariant

The sublayer output and its residual input must have the same shape, normally $[B,L,d_{\text{model}}]$. If a sublayer uses a different internal width, its final projection must return to $d_{\text{model}}$ before addition.

### 8. Padding masks

Padding positions should be masked as keys so real queries cannot retrieve padding values. Padded query positions should also be excluded from meaningful outputs and from the training loss; otherwise the model can learn from artificial positions. In cross-attention, the source padding mask applies to the encoder key/value positions.

### 9. Two useful numerical checks

First, verify that every unmasked attention row sums to approximately one and every masked future weight is approximately zero. Second, use a tiny hand-computable example and compare the implementation with $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. Shape assertions, finite-value checks, and the identical-values test are additional low-cost checks.

### 10. Acceptable versus behavior-changing simplifications

A toy implementation can use one head, tiny dimensions, fixed sequences, random or hand-designed vectors, and no training. It should still implement the score calculation, correct scaling, row-wise softmax, and weighted value aggregation. For a decoder, omitting the causal mask changes the learning problem. For a full Transformer, omitting positional information, multi-head projections, residual/normalization blocks, or feed-forward sublayers means the result is a teaching approximation rather than the paper’s architecture.

## Round 5 — Application-level reasoning and trade-offs

### 1. Decoder self-attention versus encoder-decoder attention in translation

Decoder self-attention summarizes the already generated target prefix and supports target-side language modeling. Encoder-decoder attention retrieves relevant information from the source representation for the next target word. The first answers “what target context have I generated?”; the second answers “which source information is relevant now?”

### 2. Long-distance pronoun resolution

The query at the pronoun position can assign high compatibility to a distant noun’s key and retrieve information from that noun’s value in one attention layer. This avoids requiring the information to traverse every intervening token as it would in a sequential recurrence. It is a capability, not a guarantee of correct resolution.

### 3. Doubling sequence length

The score matrix changes from $L \times L$ to $(2L) \times (2L)$, so the number of pairwise scores increases by a factor of four. The score tensor’s memory also grows approximately fourfold, ignoring other terms and implementation details.

### 4. Long-sequence trade-off

Global self-attention provides direct access to all positions and highly parallel training, but its pairwise score and memory costs are quadratic in sequence length. For very long inputs, the cost can dominate. Restricting attention to local or sparse patterns can reduce cost but changes which dependencies are directly accessible.

### 5. Investigating source copying

Encoder-decoder attention is the most direct place to inspect because its keys and values come from the source sequence. However, a high attention weight is not proof that the model copied a token: the value transformation, residual stream, feed-forward layers, output projection, and decoder state all contribute to the final prediction. Verify behavior with controlled examples or interventions.

### 6. Diagnostic but not complete explanation

Attention weights show one routing distribution for one layer and head. They do not expose all information stored in values, residual connections, feed-forward transformations, later layers, or output projections. Different internal configurations can produce similar outputs, so attention visualizations are evidence about routing, not a complete causal explanation.

### 7. Removing positional encodings

The model would lose explicit information about order and could confuse sequences with the same token content in different arrangements. Translation quality should suffer on word-order-sensitive constructions because attention could identify content but not reliably distinguish positions.

### 8. What BLEU and training-time comparisons support

BLEU supports the paper’s empirical claim that the reported Transformer systems achieved strong translation quality on the evaluated WMT tasks. Training-time comparisons support the claim that the architecture was more efficient or parallelizable than the compared systems under those experimental conditions. They do not prove that attention is universally superior for every task, sequence length, model size, or metric, nor do they isolate every causal contribution of every component.

## Round 6 — Complex scenarios and interview-level synthesis

### 1. Pronoun scenario

The representation at “it” could attend directly to “animal” or another contextually relevant token, allowing information about the antecedent to enter its contextual representation. To claim successful resolution, inspect the actual prediction or translation, evaluate controlled examples, compare alternatives, and ideally use causal interventions or ablations. An attention map alone is insufficient evidence.

### 2. Future-token leakage

This is a causal-mask bug. If target position 4 can see position 5, the model can use information from the future answer while being trained, making the objective artificially easy and creating a train-inference mismatch. The model may appear strong under teacher forcing but fail when future tokens are unavailable.

### 3. Nearly identical outputs at every position

Possible causes include collapsed or identical input representations, incorrect or shared projections, a bug broadcasting one query across positions, values that are identical, a softmax implementation producing nearly uniform or saturated weights, a mask that hides all useful distinctions, or missing positional information when content is repeated. Inspect intermediate $Q$, $K$, $V$, logits, weights, and outputs separately.

### 4. Teacher forcing versus inference

During teacher-forced training, each target position receives the ground-truth previous target tokens. During inference, it receives its own previously generated tokens, including its own mistakes. Investigate causal masking first to ensure training is valid, then examine exposure bias, decoding configuration, and whether cached autoregressive states preserve the same computation.

### 5. $d_{\text{model}}=512$ and $H=8$

The usual per-head width is:

$$
 d_k = \frac{512}{8}=64.
$$

Using $d_k=512$ independently for all eight heads greatly increases per-head projection and dot-product cost and produces a concatenated representation of width roughly $8\times512=4096$ unless another design changes it. That no longer matches the ordinary equal partition of the model width and requires a correspondingly different output projection and memory budget.

### 6. No recurrence does not mean parallel generation

The precise claim is that Transformer training can process all positions in a layer in parallel, with decoder masking preserving causality. Autoregressive inference still generates one target token at a time because token $t+1$ depends on the generated result at $t$. The architecture removes recurrent computation from the layer design; it does not remove the sequential dependency of autoregressive decoding.

### 7. A head focusing on punctuation

That pattern alone does not prove the head is useless. Punctuation may correlate with boundaries, syntax, formatting, or useful segmentation. Test its contribution through ablation or causal masking, compare performance across examples, inspect its values and downstream effects, and evaluate whether another head or layer compensates for its removal.

### 8. Debugging NaNs in attention

First check finiteness after each stage: projected $Q$, $K$, $V$, logits, masked logits, exponentials, weights, and outputs. Likely locations include overflow from an unstable softmax, excessively large unscaled dot products, invalid inputs, or a mask that makes an entire row $-\infty$. Confirm the score shape and scaling denominator, use max-subtracted softmax or a framework-stable implementation, and ensure every valid query has at least one unmasked key. Do not merely replace NaNs after the fact; locate and correct the numerical cause.

## Source and interpretation boundary

The core equation, Q/K/V roles, multi-head structure, encoder-decoder architecture, masking purpose, positional encoding, and translation/parallelism claims come from the paper, especially Sections 3.1–3.5 and the experimental discussion. Tensor layouts, debugging procedures, and some scenario recommendations are implementation guidance derived from those concepts. Interpretations such as “one head may focus on syntax” or “attention can help pronoun resolution” describe plausible behavior, not guarantees made by the paper.
