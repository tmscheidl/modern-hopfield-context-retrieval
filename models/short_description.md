# Short Description

### Cross-Attention Improvements

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
