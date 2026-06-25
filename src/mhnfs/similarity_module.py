import torch
import math
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F


class SimilarityModule(nn.Module):
    """
    Multi-head similarity module with:
    - optional L2 normalization
    - multi-head splitting
    - learnable linear projections
    - masking support
    - positive/negative weighting
    - top-k similarity selection
    - aggregation: sum / softmax / logsumexp
    - temperature scaling
    """

    def __init__(self, cfg: OmegaConf, input_dim: int = None):
        super().__init__()
        self.cfg = cfg.model.similarityModule

        self.num_heads = getattr(self.cfg, "numHeads", 1)
        self.temperature = getattr(self.cfg, "temperature", 1.0)
        self.aggregation = getattr(self.cfg, "aggregation", "sum")
        self.pos_weight = getattr(self.cfg, "posWeight", 1.0)
        self.neg_weight = getattr(self.cfg, "negWeight", 1.0)
        self.topk = getattr(self.cfg, "topk", None)

        # Optional learnable projections
        if input_dim is not None:
            assert input_dim % self.num_heads == 0, \
                "input_dim must be divisible by num_heads"

            self.query_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.support_proj = nn.Linear(input_dim, input_dim, bias=False)
        else:
            self.query_proj = None
            self.support_proj = None

    def forward(
        self,
        query_embedding: torch.Tensor,
        support_set_embeddings: torch.Tensor,
        padding_mask: torch.Tensor,
        support_set_size: torch.Tensor = None,
    ) -> torch.Tensor:

        B, _, D = query_embedding.shape
        _, N, _ = support_set_embeddings.shape

        assert D % self.num_heads == 0, "D must be divisible by num_heads"
        d_head = D // self.num_heads

        # -------------------------------
        # Assertions
        # -------------------------------
        assert query_embedding.dim() == 3 and query_embedding.shape[1] == 1
        assert support_set_embeddings.dim() == 3
        assert padding_mask.shape == (B, N) and padding_mask.dtype == torch.bool

        if support_set_size is not None:
            valid_counts = padding_mask.sum(dim=1)
            assert torch.all(valid_counts == support_set_size), \
                "support_set_size does not match padding_mask"

        # -------------------------------
        # Optional L2 normalization
        # -------------------------------
        if getattr(self.cfg, "l2Norm", False):
            query_embedding = F.normalize(query_embedding, dim=-1, eps=1e-8)
            support_set_embeddings = F.normalize(support_set_embeddings, dim=-1, eps=1e-8)

        # -------------------------------
        # Optional projections
        # -------------------------------
        if self.query_proj is not None:
            query_embedding = self.query_proj(query_embedding)
            support_set_embeddings = self.support_proj(support_set_embeddings)

        # -------------------------------
        # Multi-head split (SAFE reshape)
        # -------------------------------
        query = query_embedding.reshape(B, 1, self.num_heads, d_head).transpose(1, 2)
        support = support_set_embeddings.reshape(B, N, self.num_heads, d_head).transpose(1, 2)

        # -------------------------------
        # Similarity computation
        # -------------------------------
        similarities = torch.matmul(query, support.transpose(-2, -1)) / math.sqrt(d_head)

        # ✅ FIXED: correct temperature scaling
        similarities = similarities / self.temperature

        similarities = torch.nan_to_num(similarities)

        # -------------------------------
        # Masking (SAFE)
        # -------------------------------
        mask = padding_mask.unsqueeze(1).unsqueeze(2)

        # ✅ FIXED: no -inf (stable for all aggregations)
        similarities = similarities.masked_fill(~mask, 0.0)

        # -------------------------------
        # Positive / negative weighting
        # -------------------------------
        sim_pos = torch.clamp(similarities, min=0.0) * self.pos_weight
        sim_neg = torch.clamp(similarities, max=0.0) * self.neg_weight
        similarities = sim_pos + sim_neg

        # -------------------------------
        # Top-k selection
        # -------------------------------
        if self.topk is not None and self.topk > 0:
            k = min(self.topk, N)
            similarities, _ = torch.topk(similarities, k=k, dim=-1)

        # -------------------------------
        # Aggregation over support set
        # -------------------------------
        if self.aggregation == "sum":
            similarity_sums = similarities.sum(dim=-1)

        elif self.aggregation == "softmax":
            attn = torch.softmax(similarities, dim=-1)
            similarity_sums = (attn * similarities).sum(dim=-1)

        elif self.aggregation == "logsumexp":
            similarity_sums = torch.logsumexp(similarities, dim=-1)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # -------------------------------
        # Aggregate over heads
        # -------------------------------
        similarity_sums = similarity_sums.mean(dim=1)  # [B,1]

        # -------------------------------
        # Scaling
        # -------------------------------
        if self.aggregation == "sum" and support_set_size is not None:
            stabilizer = 1e-8
            N = support_set_size.reshape(-1, 1).float()

            scaling = getattr(self.cfg, "scaling", "1/N")

            if scaling == "1/N":
                similarity_sums = similarity_sums / (2.0 * N + stabilizer)

            elif scaling == "1/sqrt(N)":
                similarity_sums = similarity_sums / (
                    2.0 * torch.sqrt(N) + stabilizer
                )

        return similarity_sums

    # Double-check mask semantics
    # Add assertion checks
    # Add temperature scaling
    # Add optional softmax weighting (attention version)
    # Separate positive / negative importance
    # Replace sum with log-sum-exp
    # Top-k similarity
    # Interpret similarity as energy
    # multi-head similarity
    # Learnable projections for multi-head