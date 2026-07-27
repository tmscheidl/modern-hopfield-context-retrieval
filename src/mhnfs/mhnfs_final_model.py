import torch
import torch.nn as nn

class MHNfsFinalModel(nn.Module):
    """
    Final MHNfs model that stacks:
    CrossAttentionModule -> ContextModule -> SimilarityModule

    Computes prediction for query molecules based on active/inactive support sets.
    """

    def __init__(self, cross_attention, context_module, similarity_module,
             prediction_scaling, learnable_scaling=False):
        super().__init__()
        self.cross_attention = cross_attention
        self.context_module = context_module
        self.similarity_module = similarity_module

        if learnable_scaling:
            self.prediction_scaling = nn.Parameter(torch.tensor(float(prediction_scaling)))
        else:
            self.register_buffer("prediction_scaling", torch.tensor(float(prediction_scaling)))

    def forward(self, query, support_actives, support_inactives,
            mask_actives, mask_inactives, context_memory):
        assert query.dim() == 3 and query.shape[1] == 1

        # assign actives/inactives first
        actives = support_actives
        inactives = support_inactives

        # 1. Context Module FIRST
        query, actives, inactives = self.context_module(
            query, actives, inactives, context_memory
        )

        # 2. Cross-Attention SECOND
        query, actives, inactives = self.cross_attention(
            query, actives, inactives, mask_actives, mask_inactives
        )

        # 3. Similarity
        support_size_a = mask_actives.sum(dim=1)
        support_size_i = mask_inactives.sum(dim=1)

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

        logits = (sim_active - sim_inactive) * self.prediction_scaling
        return logits
