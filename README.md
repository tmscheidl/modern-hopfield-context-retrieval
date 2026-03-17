A modular deep learning architecture that combines:

- Modern Hopfield Networks
- Cross-Attention mechanisms
- Context-based retrieval
- Metric-based similarity learning

The model is designed for few-shot learning scenarios, where predictions
are made based on a support set of examples.

## Architecture

The pipeline consists of three main components:

1. Context Module
   - Multi-step Hopfield retrieval
   - Top-k memory selection
   - Gated residual updates

2. Hopfield Memory
   - Multi-head associative retrieval
   - Learnable temperature (β)
   - Energy-based convergence

3. Similarity Module
   - Computes prediction via similarity-weighted aggregation
