A modular deep learning architecture combining:

- Modern Hopfield Networks
- Cross-Attention Module
- Context Module
- Metric-based similarity Module

Designed for few-shot learning, the model makes predictions based on a support set of examples and is tailored to handle active and inactive molecules.

## Architecture

The pipeline consists of three main components:

Cross-Attention Module

- Converts self-attention into true cross-attention (query ≠ key/value) for better information flow.
- Uses scaled dot-product attention with dropout for stability.
- Adds active/inactive bias embeddings and gated residual connections for more precise updates.

Context Module

- Implements multi-step Hopfield retrieval to iteratively refine representations.
- Includes top-k memory selection to focus on the most relevant support examples.
- Uses learnable gating and FFN to improve feature aggregation and stability.
- Normalizes embeddings and separates projections per type for cleaner computations.

Hopfield Module

- Supports multi-head associative retrieval with learnable temperature (β).
- Includes residual connections, masking, and dropout for robustness.
- Adds energy computation and iterative retrieval to ensure convergence.
- Improved initialization and numerical stability for reliable training.

Similarity Module

Computes predictions via similarity-weighted aggregation of support set labels.

This design ensures efficient, stable, and accurate few-shot predictions by combining retrieval-based memory, attention, and similarity scoring.
