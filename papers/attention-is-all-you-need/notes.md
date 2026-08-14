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

## Interaction 2 — Commit and push paper notes
**Q:** Commit and push the Attention Is All You Need workspace changes to the remote repository.
**A:** Inspected the active notes and Git state. The repository is on `main`, tracking `origin/main`; the focused changes are the new Transformer paper workspace and linked knowledge indexes. The requested changes will be committed and pushed to `origin/main`.
