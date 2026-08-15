# Questions: Attention Is All You Need

- How does the scaled dot-product attention equation produce one output vector for each token?
- Why does dividing by $\sqrt{d_k}$ stabilize softmax attention?
- How exactly does the decoder mask prevent information leakage during training?
- What is the tensor shape of $Q$, $K$, $V$, attention scores, and outputs in a batched implementation?
- Which simplifications are acceptable in a small prototype, and which would change the Transformer behavior?


## Active diagnostic — Round 1: Foundations
**Status:** Awaiting learner responses
**Scope:** Paper Sections 3.2 and 3.5, plus the encoder-decoder architecture in Section 3.1.

1. What problem does attention solve that is difficult for a plain RNN, especially for long sequences?
2. Explain the roles of $Q$, $K$, and $V$. Why are keys and values conceptually different?
3. In self-attention, where do the queries, keys, and values come from? Why can two tokens in the same sentence produce different output representations?
4. Given one query's scaled scores $[2,1,0]$, which token receives the greatest softmax weight, and what happens when those weights multiply $V$?
5. Why does scaled dot-product attention divide by $\sqrt{d_k}$? What could happen if scaling were omitted when $d_k$ is large?
6. For one head with sequence length $L=5$, $d_k=64$, and $d_v=32$, what are the shapes of $Q$, $K$, $V$, $QK^\top$, and the final attention output?
7. Why does attention need positional encoding? What ambiguity exists if the same token embeddings are supplied in a different order?
8. In decoder encoder-decoder attention, which sequence supplies the queries, and which sequence supplies the keys and values?


## Round 2 — Core mechanics and numerical reasoning
**Status:** Not started
**Scope:** Paper Section 3.2 and the attention-from-scratch derivation.

1. For $q=[2,1]$, $k_1=[0,1]$, $k_2=[2,1]$, and $k_3=[1,0]$, calculate the raw dot-product scores, the scaled scores for $d_k=2$, and identify the largest score.
2. Why is softmax applied independently to each row of $QK^\top$ rather than to the entire score matrix?
3. Explain how the matrix product $\operatorname{softmax}(QK^\top / \sqrt{d_k})V$ produces one output vector for every query position.
4. For batch size $B=2$, heads $H=8$, sequence length $L=16$, $d_k=64$, and $d_v=64$, state the shapes of $Q$, $K$, $V$, the score tensor, and the output before concatenating heads.
5. Why can $d_v$ differ from $d_k$ in principle? Which dimension determines the scaling factor?
6. State the multi-head attention computation using $Q_i$, $K_i$, and $V_i$ for head $i$. Why is a final output projection needed?
7. Construct the causal visibility pattern for a four-token decoder input. Which positions may token 3 attend to during training?
8. In encoder-decoder attention with source length $L_s$ and target length $L_t$, what are the sources of $Q$, $K$, and $V$, and what is the score shape per head?
9. If every value vector in an attention head is identical, what will the weighted-sum output be for each query, regardless of the attention weights? Explain why.

## Round 3 — Transformer architecture and training behavior
**Status:** Not started
**Scope:** Paper Sections 3.1, 3.3, 3.4, and 3.5.

1. Describe the information flow through one encoder layer and one decoder layer in the original Transformer.
2. What is the difference between encoder self-attention, decoder masked self-attention, and encoder-decoder attention?
3. Why can the decoder’s target positions be processed in parallel during training but not normally during autoregressive inference?
4. What information does the causal mask hide, and why is masking necessary even when the target sequence is already shifted during teacher forcing?
5. What do residual connections and layer normalization contribute to optimization and representation flow?
6. Why is the feed-forward sublayer called position-wise? Does it mix information between token positions by itself?
7. Why are positional encodings added to token embeddings? What role do the original sinusoidal encodings play?
8. What is the key parallelism advantage of self-attention over recurrence, and what long-range dependency advantage does the paper associate with it?
9. Why does multi-head attention generally provide more expressive routing than one attention operation with the same total model width?
10. Which parts of the original Transformer are responsible for source understanding, target-side language modeling, and source-target alignment?

## Round 4 — Implementation and debugging
**Status:** Not started
**Scope:** Paper equations translated into tensor operations; implementation details are study extensions inferred from the paper’s shapes.

1. Starting from input shape $[B,L,d_{\text{model}}]$, describe the reshape and transpose steps needed to obtain per-head tensors of shape $[B,H,L,d_k]$ when $d_{\text{model}}=H d_k$.
2. A program computes $QK^\top$ with shape $[B,H,d_k,L]$. What transpose or layout mistake is likely, and what shape should the scores have?
3. Where should a causal mask be applied relative to softmax, and what value should masked logits effectively receive?
4. Why is subtracting the row maximum before exponentiating a useful softmax implementation technique?
5. A developer divides attention scores by $\sqrt{L}$ instead of $\sqrt{d_k}$. What is wrong, and when might this bug be hard to notice?
6. After attention, why must the heads be transposed, concatenated, and passed through an output projection before the residual addition?
7. What shape invariant must hold before adding a sublayer output to its residual input?
8. How should padding positions be prevented from receiving or contributing attention in a batch containing different sequence lengths?
9. Give two small numerical checks that can catch an attention implementation bug before training a full model.
10. Which simplifications are acceptable in a toy attention prototype, and which omissions would change the defining behavior of the Transformer?

## Round 5 — Application-level reasoning and trade-offs
**Status:** Not started
**Scope:** Translation use case, paper claims, and careful interpretation of attention behavior.

1. In machine translation, explain what information is retrieved by decoder self-attention versus encoder-decoder attention when generating a target word.
2. Why is direct attention useful for resolving a long-distance relation such as a pronoun referring to an earlier noun?
3. If sequence length grows from $L$ to $2L$, how do the number of pairwise attention scores and the approximate score-memory cost change?
4. What trade-off is created by replacing recurrence with global self-attention for very long sequences?
5. If a decoder appears to copy a source phrase, which attention mechanism is the most direct place to investigate, and why should this not be treated as proof of copying?
6. Why can attention weights be useful diagnostics without being complete explanations of the model’s reasoning?
7. How would removing positional encodings affect a translation system, and why might the damage be especially visible for word order?
8. The paper reports translation quality using BLEU and training-time comparisons. What do those measurements support, and what do they not prove about attention in general?

## Round 6 — Complex scenarios and interview-level synthesis
**Status:** Not started
**Scope:** Integrated reasoning across the paper; distinguish paper claims from interpretation and implementation inference.

1. In “The animal did not cross the street because it was tired,” how could self-attention help the representation of “it,” and what would you need to inspect before claiming that the model resolved the pronoun correctly?
2. During training, a decoder predicts target token 4. A bug lets its query attend to target token 5. What kind of failure is this, and how would it affect the training objective?
3. A model produces nearly identical contextual outputs for every position even though the input tokens differ. Give several possible causes involving projections, softmax, values, masking, or positional information.
4. A model trains well with teacher forcing but generates poor sequences at inference time. Which difference between training and autoregressive decoding should you investigate first?
5. A Transformer has $d_{\text{model}}=512$ and $H=8$. What is the usual per-head $d_k$? If the implementation instead uses $d_k=512$ for every head without changing the output design, what computational and architectural issue arises?
6. An engineer claims that because the Transformer has no recurrence, generation should be fully parallel. Correct the claim precisely for training and inference.
7. Attention maps show a head concentrating on punctuation. Does that alone demonstrate that the head is useless? What additional evidence would you seek?
8. Design a minimal end-to-end debugging plan for a Transformer whose attention scores contain NaNs. Identify the likely numerical location, immediate checks, and one safe corrective measure.

## Suggested progression
Study and answer the rounds in order, but do not treat them as a required curriculum. Use `answers.md` for self-checking after attempting each round. The rounds separately exercise intuition, mathematics, architecture, implementation, application, and integrated reasoning.
