#This is the original MHNfs Similarity Module

import torch
from omegaconf import OmegaConf


def similarity_module(
    query_embedding: torch.Tensor,
    support_set_embeddings: torch.Tensor,
    padding_mask: torch.Tensor,
    support_set_size: torch.Tensor,
    cfg: OmegaConf,
) -> torch.Tensor:
    """
    Similarity Module

    Computes activity prediction for a query by aggregating similarities
    with support set embeddings.

    This module is applied twice:
    - active support set
    - inactive support set

    Args:
        query_embedding: [B, 1, D]
        support_set_embeddings: [B, N, D]
        padding_mask: [B, N] (True = valid, False = padding)
        support_set_size: [B]
        cfg: Hydra config

    Returns:
        similarity score: [B, 1]
    """

    # -------------------------------------------------
    # Optional L2 normalization (cosine similarity)
    # -------------------------------------------------
    if cfg.model.similarityModule.l2Norm:
        query_embedding = torch.nn.functional.normalize(
            query_embedding, dim=-1, eps=1e-8
        )
        support_set_embeddings = torch.nn.functional.normalize(
            support_set_embeddings, dim=-1, eps=1e-8
        )

    # -------------------------------------------------
    # Similarity computation
    # -------------------------------------------------
    similarities = torch.matmul(
        query_embedding,
        support_set_embeddings.transpose(1, 2)
    )  # [B, 1, N]

    # -------------------------------------------------
    # Mask padding
    # -------------------------------------------------
    mask = padding_mask.bool().unsqueeze(1)  # [B,1,N]

    similarities = torch.nan_to_num(similarities)
    similarities = similarities * mask

    # -------------------------------------------------
    # Aggregate similarities
    # -------------------------------------------------
    similarity_sums = similarities.sum(dim=2)  # [B,1]

    # -------------------------------------------------
    # Scaling
    # -------------------------------------------------
    stabilizer = 1e-8
    N = support_set_size.reshape(-1, 1).float()

    if cfg.model.similarityModule.scaling == "1/N":
        similarity_sums = similarity_sums / (2.0 * N + stabilizer)

    elif cfg.model.similarityModule.scaling == "1/sqrt(N)":
        similarity_sums = similarity_sums / (2.0 * torch.sqrt(N) + stabilizer)

    return similarity_sums