import torch
import math
from omegaconf import OmegaConf
import torch.nn as nn
import torch.nn.functional as F


class SimilarityModule(nn.Module):
    """
    Multi-head similarity module.

    The module:
    - optionally normalizes query and support representations
    - optionally projects representations into separate spaces
    - splits representations into multiple heads
    - computes query-support similarities
    - applies padding masks and positive/negative weighting
    - optionally selects the most similar support molecules
    - aggregates similarities using sum, softmax, or logsumexp
    - applies temperature and support-set-size scaling
    """

    def __init__(self, cfg: OmegaConf, input_dim: int = None):
        super().__init__()
        self.cfg = cfg.model.similarityModule

        # Similarity configuration.
        self.num_heads = getattr(self.cfg, "numHeads", 1)
        self.temperature = getattr(self.cfg, "temperature", 1.0)
        self.aggregation = getattr(self.cfg, "aggregation", "sum")
        self.pos_weight = getattr(self.cfg, "posWeight", 1.0)
        self.neg_weight = getattr(self.cfg, "negWeight", 1.0)
        self.topk = getattr(self.cfg, "topk", None)

        # Optional learnable projections.
        # Separate projections can transform query and support
        # representations before calculating their similarity.
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

        # Read the batch size, support-set size, and embedding dimension.
        B, _, D = query_embedding.shape
        _, N, _ = support_set_embeddings.shape

        # Each head receives an equal part of the embedding dimension.
        assert D % self.num_heads == 0, "D must be divisible by num_heads"
        d_head = D // self.num_heads

        # -------------------------------
        # Input validation
        # -------------------------------
        # Verify that the query contains exactly one molecule per sample,
        # and that the support embeddings and padding mask have compatible shapes.
        assert query_embedding.dim() == 3 and query_embedding.shape[1] == 1
        assert support_set_embeddings.dim() == 3
        assert padding_mask.shape == (B, N) and padding_mask.dtype == torch.bool

        # If the support-set size is provided, verify that it matches
        # the number of valid entries in the padding mask.
        if support_set_size is not None:
            valid_counts = padding_mask.sum(dim=1)
            assert torch.all(valid_counts == support_set_size), \
                "support_set_size does not match padding_mask"

        # -------------------------------
        # Optional L2 normalization
        # -------------------------------
        # Normalize representations to unit length before calculating
        # similarity. This makes the dot product equivalent to cosine similarity.
        if getattr(self.cfg, "l2Norm", False):
            query_embedding = F.normalize(query_embedding, dim=-1, eps=1e-8)
            support_set_embeddings = F.normalize(support_set_embeddings, dim=-1, eps=1e-8)

        # -------------------------------
        # Optional projections
        # -------------------------------
        # Apply separate learnable projections to query and support
        # representations when projection layers are enabled.
        if self.query_proj is not None:
            query_embedding = self.query_proj(query_embedding)
            support_set_embeddings = self.support_proj(support_set_embeddings)

        # -------------------------------
        # Multi-head split
        # -------------------------------
        # Split each representation into multiple independent heads.
        # Each head operates on a smaller subspace of the embedding.
        query = query_embedding.reshape(B, 1, self.num_heads, d_head).transpose(1, 2)
        support = support_set_embeddings.reshape(B, N, self.num_heads, d_head).transpose(1, 2)

        # -------------------------------
        # Similarity computation
        # -------------------------------
        # Calculate the dot-product similarity between the query
        # and every support molecule for each attention head.
        similarities = torch.matmul(query, support.transpose(-2, -1))

        # Temperature controls the scale of the similarity values.
        similarities = similarities / self.temperature

        # Replace possible NaN or infinite values with finite values
        # to keep subsequent operations numerically stable.
        similarities = torch.nan_to_num(similarities)

        # -------------------------------
        # Masking
        # -------------------------------
        # Expand the support mask to match the head and query dimensions.
        mask = padding_mask.unsqueeze(1).unsqueeze(2)

        # Invalid/padded support molecules contribute zero similarity.
        similarities = similarities.masked_fill(~mask, 0.0)

        # -------------------------------
        # Positive / negative weighting
        # -------------------------------
        # Positive similarities and negative similarities can have
        # different importance through separate learnable/configured weights.
        sim_pos = torch.clamp(similarities, min=0.0) * self.pos_weight
        sim_neg = torch.clamp(similarities, max=0.0) * self.neg_weight
        similarities = sim_pos + sim_neg

        # -------------------------------
        # Top-k selection
        # -------------------------------
        # Optionally keep only the k most similar support molecules.
        if self.topk is not None and self.topk > 0:
            k = min(self.topk, N)
            similarities, _ = torch.topk(similarities, k=k, dim=-1)

        # -------------------------------
        # Aggregation over support set
        # -------------------------------
        # Combine the individual query-support similarities into
        # one similarity score for each attention head.
        if self.aggregation == "sum":
            similarity_sums = similarities.sum(dim=-1)

        elif self.aggregation == "softmax":
            # Use similarity values as attention scores and calculate
            # a weighted average that emphasizes more similar supports.
            attn = torch.softmax(similarities, dim=-1)
            similarity_sums = (attn * similarities).sum(dim=-1)

        elif self.aggregation == "logsumexp":
            # Smoothly aggregate support similarities while emphasizing
            # the strongest similarity values.
            similarity_sums = torch.logsumexp(similarities, dim=-1)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # -------------------------------
        # Aggregate over heads
        # -------------------------------
        # Average the similarity scores from all heads.
        similarity_sums = similarity_sums.mean(dim=1)  # [B,1]

        # -------------------------------
        # Support-set-size scaling
        # -------------------------------
        # For sum aggregation, normalize the score according to
        # the number of valid support molecules.
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
    # Multi-head similarity
    # Learnable projections for multi-head
