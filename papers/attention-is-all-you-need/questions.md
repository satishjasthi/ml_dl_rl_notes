# Questions: Attention Is All You Need

- How does the scaled dot-product attention equation produce one output vector for each token?
- Why does dividing by $\sqrt{d_k}$ stabilize softmax attention?
- How exactly does the decoder mask prevent information leakage during training?
- What is the tensor shape of $Q$, $K$, $V$, attention scores, and outputs in a batched implementation?
- Which simplifications are acceptable in a small prototype, and which would change the Transformer behavior?
