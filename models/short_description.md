# Short Description

## Cross-Attention Improvements

Compared to the original implementation, the Cross-Attention module was redesigned to improve **stability, efficiency, and expressiveness**.

**Changes:**
1. **Self-Attention → True Cross-Attention (Q ≠ K,V)**  
   → The query now attends explicitly to support molecules, improving retrieval quality  
2. **Manual Attention → scaled_dot_product_attention**  
   → Uses optimized PyTorch implementation for better speed and numerical stability  
3. **Soft MoE Routing (Top-1 at inference)**  
   → Improves training by allowing smoother expert selection while keeping efficient inference  
4. **Attention Dropout**  
   → Reduces overfitting and improves generalization  
5. **Active / Inactive Bias Embeddings**  
   → Helps the model clearly distinguish molecule types  
6. **Residual Scaling (DeepNorm-lite)**  
   → Stabilizes deep updates and prevents exploding activations  
7. **Gated Residual Connections (optional)**  
   → Allows the model to control how much new information is added  
8. **Additional small stability improvements**  
   → Better normalization and safer training behavior  

**Result:**  
A more **stable, efficient, and task-aware attention mechanism**, better suited for few-shot molecular classification.


## Context Module Improvements

The original implementation performed a **single-step Hopfield retrieval over all embeddings**, using a simple residual update.  
The improved version introduces a more **structured and stable retrieval process**.

**Changes:**
1. **Learnable gating**  
   → Controls how much retrieved information is added, preventing over-updates  
2. **Embedding normalization (LayerNorm + L2)**  
   → Improves numerical stability and consistency across retrieval steps  
3. **Separate projections for query / active / inactive**  
   → Allows more specialized representations for different molecule types  
4. **Learnable temperature (β)**  
   → Enables adaptive control of Hopfield retrieval sharpness  
5. **Feed-Forward Network (FFN) after retrieval**  
   → Refines representations similar to Transformer blocks  
6. **Multi-step retrieval**  
   → Iteratively improves representations instead of a single update  
7. **Top-k memory selection**  
   → Focuses on the most relevant context molecules, reducing noise  

**Result:**  
A more **stable, focused, and iterative context retrieval mechanism**, improving representation quality for few-shot learning.

## Similarity Module

The original implementation computed similarity between a query and a support set using a simple dot-product, optionally normalized.  
The improved version introduces a **multi-head, flexible, and numerically stable similarity computation**.

**Changes:**
1. **Multi-head similarity**  
   → Splits embeddings into multiple heads for richer, parallel similarity comparisons
2. **Optional learnable linear projections per head**  
   → Allows each head to specialize its representation for better alignment
3. **Optional L2 normalization**  
   → Stabilizes magnitude differences across embeddings
4. **Masking support**  
   → Safely ignores padding or inactive support vectors
5. **Positive / negative weighting**  
   → Adjusts the influence of positive vs negative similarity values
6. **Top-k similarity selection**  
   → Focuses on the most relevant support vectors, reducing noise
7. **Aggregation strategies: sum / softmax / log-sum-exp**  
   → Flexible combination of similarities depending on downstream needs
8. **Temperature scaling**  
   → Controls sensitivity of similarity scores
9. **Energy interpretation**  
   → Higher similarity → lower “energy”, aligning with Hopfield-style reasoning

**Result:**  
A robust, flexible similarity module suitable for few-shot learning and multi-head embedding comparisons, supporting advanced weighting, masking, and aggregation strategies.
