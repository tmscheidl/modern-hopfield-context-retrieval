import torch
import torch.nn as nn

class MHNfsFinalModel(nn.Module):
    """
    Final MHNfs model that stacks:
    CrossAttentionModule -> ContextModule -> SimilarityModule

    Computes prediction for query molecules based on active/inactive support sets.
    """

    def __init__(self, cross_attention: nn.Module,
                 context_module: nn.Module,
                 similarity_module: nn.Module):
        super().__init__()
        self.cross_attention = cross_attention
        self.context_module = context_module
        self.similarity_module = similarity_module

    def forward(self,
                query,
                support_actives,
                support_inactives,
                mask_actives,
                mask_inactives,
                context_memory):
        """
        Args:
            query: [B, 1, D]
            support_actives: [B, N_a, D]
            support_inactives: [B, N_i, D]
            mask_actives: [B, N_a] (True = valid)
            mask_inactives: [B, N_i] (True = valid)
            context_memory: [B, Nc, D]

        Returns:
            logits: [B,1] (higher → more likely active)
        """

        # -------------------------
        # Safety checks
        # -------------------------
        assert query.dim() == 3 and query.shape[1] == 1, \
            "Query must have shape [B,1,D]"

        # -------------------------
        # 1. Cross-Attention
        # -------------------------
        query, actives, inactives = self.cross_attention(
            query,
            support_actives,
            support_inactives,
            mask_actives,
            mask_inactives
        )

        # -------------------------
        # 2. Context Module (Hopfield)
        # -------------------------
        query, actives, inactives = self.context_module(
            query,
            actives,
            inactives,
            context_memory
        )

        # -------------------------
        # 3. Prepare support sizes
        # -------------------------
        support_size_a = mask_actives.sum(dim=1)
        support_size_i = mask_inactives.sum(dim=1)

        # -------------------------
        # 4. Similarity Module
        # -------------------------
        sim_active = self.similarity_module(
            query_embedding=query,
            support_set_embeddings=actives,
            padding_mask=mask_actives,
            support_set_size=support_size_a
        )

        sim_inactive = self.similarity_module(
            query_embedding=query,
            support_set_embeddings=inactives,
            padding_mask=mask_inactives,
            support_set_size=support_size_i
        )

        # -------------------------
        # 5. Final prediction (logits)
        # -------------------------
        logits = sim_active - sim_inactive

        return logits