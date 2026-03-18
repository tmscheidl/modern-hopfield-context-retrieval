
## Hopfield Memory (Improved)

The original Hopfield module was redesigned into a **Transformer-like Hopfield layer**.  
Instead of using a classical or older Hopfield formulation, the new version follows a **modern attention-based architecture**, making it closer to a Transformer while still preserving the energy-based memory retrieval of Hopfield networks.

### What was changed

- **Transformer-style attention (Q, K, V projections)**  
  The module now uses query, key, and value projections, similar to Transformers, enabling more flexible and expressive memory access.

- **Multi-head retrieval**  
  Multiple heads allow the model to retrieve different patterns in parallel, increasing representation power.

- **Learnable temperature (β)**  
  Controls the sharpness of attention. This replaces fixed scaling and makes retrieval adaptive.

- **Residual connections**  
  Stabilize training and keep original information, similar to Transformer layers.

- **Attention dropout**  
  Improves generalization and reduces overfitting.

- **Mask support**  
  Ensures padding or invalid entries do not affect retrieval.

- **Iterative retrieval (energy-based)**  
  The state can be updated multiple times until convergence, which is not present in standard Transformers.

- **Energy function**  
  Introduces a Hopfield-specific concept to measure convergence and stability of memory retrieval.

- **Improved initialization & numerical stability**  
  Prevents unstable training and extreme values.

### Why it is not a standard Transformer

Although the module uses Transformer-like attention, it is **not a standard Transformer** because:

- It is based on **energy minimization (Hopfield networks)** rather than purely feed-forward attention.
- It supports **iterative updates until convergence**, instead of a single forward pass.
- It includes an explicit **energy function** to analyze retrieval.
- The attention acts as **associative memory retrieval**, not just feature mixing.

### How it works (short)

1. The input is projected into query, key, and value representations.
2. Similarities between query and stored patterns (keys) are computed.
3. Attention weights are calculated using a **learnable temperature (β)**.
4. The output is a weighted sum over values (memory retrieval).
5. The result is added back to the input (residual update).
6. Optionally, this process is repeated iteratively until convergence.

This results in a **memory-augmented Transformer layer** that combines attention with associative retrieval.
