A modular deep learning architecture combining:

- Modern Hopfield Networks
- Cross-Attention Module
- Context Module
- Metric-based similarity Module

Designed for few-shot learning, the model makes predictions based on a support set of examples and is tailored to handle active and inactive molecules.

## Architecture

The pipeline consists of four main components:

### Cross-Attention Module
- Uses true cross-attention to enable interaction between query and support set.
- Improves stability and expressiveness with modern attention mechanisms and residual connections.
- Incorporates task-specific biases for active/inactive molecules.

### Context Module
- Refines representations through iterative memory retrieval.
- Focuses on the most relevant support examples.
- Enhances feature aggregation and stability with gating and normalization.

### Hopfield Module
- Implements a Transformer-like associative memory mechanism.
- Uses multi-head attention with learnable temperature for flexible retrieval.
- Extends standard Transformers with energy-based and iterative retrieval.

### Similarity Module
- Computes predictions based on similarity between query and support set.

---

This architecture combines attention, memory retrieval, and similarity learning to enable stable and effective few-shot predictions for active/inactive molecules.

---

References:
- https://github.com/ml-jku/MHNfs/tree/main/src/mhnfs
- https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
- https://openreview.net/pdf?id=XrMWUuEevr
