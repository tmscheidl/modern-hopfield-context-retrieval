import torch
import torch.nn as nn

class MHNfsFinalModel(nn.Module):
    """
    Final MHNfs model that combines three processing stages:

    Context Module -> Cross-Attention Module -> Similarity Module

    The model predicts the activity of a query molecule by comparing it
    with the active and inactive support molecules.
    """

    def __init__(self, cross_attention, context_module, similarity_module,
             prediction_scaling, learnable_scaling=False):
        super().__init__()
        self.cross_attention = cross_attention
        self.context_module = context_module
        self.similarity_module = similarity_module

        # Use either a fixed scaling factor or a learnable scaling parameter
        # to control the magnitude of the final prediction logits.
        if learnable_scaling:
            self.prediction_scaling = nn.Parameter(torch.tensor(float(prediction_scaling)))
        else:
            self.register_buffer("prediction_scaling", torch.tensor(float(prediction_scaling)))

    def forward(self, query, support_actives, support_inactives,
            mask_actives, mask_inactives, context_memory):
        # The query must contain exactly one molecule per task.
        assert query.dim() == 3 and query.shape[1] == 1

        # Keep the active and inactive support sets separate.
        actives = support_actives
        inactives = support_inactives

        # 1. Context Module FIRST
        # Enrich the query and support representations using
        # information retrieved from the context memory.
        query, actives, inactives = self.context_module(
            query, actives, inactives, context_memory
        )

        # 2. Cross-Attention SECOND
        # Model interactions between the query and the support molecules
        # using unified self-attention.
        query, actives, inactives = self.cross_attention(
            query, actives, inactives, mask_actives, mask_inactives
        )

        # 3. Similarity
        # Count the valid molecules in each support group.
        support_size_a = mask_actives.sum(dim=1)
        support_size_i = mask_inactives.sum(dim=1)

        # Calculate the similarity between the query and active supports.
        sim_active = self.similarity_module(
            query_embedding=query,
            support_set_embeddings=actives,
            padding_mask=mask_actives,
            support_set_size=support_size_a
        )

        # Calculate the similarity between the query and inactive supports.
        sim_inactive = self.similarity_module(
            query_embedding=query,
            support_set_embeddings=inactives,
            padding_mask=mask_inactives,
            support_set_size=support_size_i
        )

        # The final prediction is based on the difference between
        # active and inactive similarity scores.
        logits = (sim_active - sim_inactive) * self.prediction_scaling
        return logits
