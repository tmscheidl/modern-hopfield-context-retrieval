# Short Description

### Cross-Attention Module (Improved)

The original implementation used **self-attention over concatenated inputs**, where query and support examples were processed together.  
This was replaced with **true cross-attention (Q ≠ K,V)**, allowing the query to directly attend to support molecules.

**Key improvements:**
- **True cross-attention** → better separation between query and support  
- **Scaled dot-product attention** → faster and more stable computation  
- **Active / inactive bias embeddings** → explicit distinction between molecule types  
- **Learnable temperature** → adaptive control of attention sharpness  
- **RMSNorm + gated residuals** → improved training stability  
- **Mixture-of-Experts (MoE)** → higher model capacity and flexibility  

Overall, these changes make the module more **stable, expressive, and effective for few-shot molecular classification**.

