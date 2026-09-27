import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

PROJECT_ROOT = "/system/user/studentwork/tscheidl/MHNfs"
sys.path.append(PROJECT_ROOT)

from src.mhnfs.hopfield.my_hopfield import MyHopfield

#from mhnfs.modules import Hopfield # this is the original

# -------------------------------------------------
# Weight initialization
# -------------------------------------------------
def init_weights(module_type, module):
    # Initialize linear layers with Xavier initialization.
    # Biases are initialized to zero.
    if module_type == "linear" and isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# -------------------------------------------------
# ContextModule
# -------------------------------------------------
class ContextModule(nn.Module):
    """
    Context module based on iterative Modern Hopfield retrieval.

    The module:
    - selects the most relevant context molecules
    - retrieves contextual information using a Hopfield layer
    - updates query and support representations using gated residuals
    - refines the representations with a Transformer-style FFN
    """

    def __init__(self, cfg, top_k=None):
        super().__init__()

        dim = cfg.model.associationSpace_dim
        ffn_mult = 4

        self.num_steps = cfg.model.hopfield.num_steps
        #self.top_k = top_k
        # Number of context molecules used for Hopfield retrieval.
        # The configured value is used unless a different value is provided.
        self.top_k = top_k if top_k is not None else getattr(cfg.model, "context_top_k", 512)

        # -------------------------------------------------
        # Hopfield Memory
        # -------------------------------------------------
        # Modern Hopfield layer used to retrieve relevant
        # information from the selected context molecules.
        self.hopfield = MyHopfield(
            input_size=dim,
            num_heads=cfg.model.hopfield.heads,
            init_beta=cfg.model.hopfield.beta,
            attn_dropout=cfg.model.hopfield.dropout,
            beta_min=getattr(cfg.model.hopfield, "beta_min", 0.001),
        )
        self.hopfield.apply(partial(init_weights, "linear"))

        # -------------------------------------------------
        # Projections
        # -------------------------------------------------
        # Project the query and the two support groups into
        # separate representations before Hopfield retrieval.
        self.query_proj = nn.Linear(dim, dim)
        self.active_proj = nn.Linear(dim, dim)
        self.inactive_proj = nn.Linear(dim, dim)

        # -------------------------------------------------
        # Gates
        # -------------------------------------------------
        # Learnable gates control how strongly the retrieved
        # representations are applied to the original states.
        self.query_gate = nn.Parameter(torch.full((dim,), -0.5))
        self.support_gate = nn.Parameter(torch.full((dim,), -1.0))

        # -------------------------------------------------
        # Normalization
        # -------------------------------------------------
        # Normalize representations before retrieval and FFN
        # refinement. Affine parameters are disabled.
        self.pre_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False)

        # -------------------------------------------------
        # Feed Forward Network
        # -------------------------------------------------
        # Transformer-style FFN that temporarily expands the
        # representation dimension by a factor of four.
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(0.5),
        )

    # -------------------------------------------------
    # L2 normalization
    # -------------------------------------------------
    def l2_norm(self, x):
        # Normalize each representation to unit length.
        # This is useful when comparing representations by similarity.
        return F.normalize(x, dim=-1, eps=1e-8)

    # -------------------------------------------------
    # Top-K context selection
    # -------------------------------------------------
    def topk_context(self, query, context):
        """
        Select the context molecules most similar to the query.

        query:
            [B, 1, D] — query molecule representation

        context:
            [Nc, D] or [B, Nc, D] — available context molecules

        return:
            [B, top_k, D] — selected context molecules
        """

        B = query.size(0)
        D = query.size(-1)

        # Convert a shared context set to a batch of context sets.
        if context.dim() == 2:
            context = context.unsqueeze(0).expand(B, -1, -1)
        elif context.dim() == 3:
            if context.size(0) == 1 and B > 1:
                context = context.expand(B, -1, -1)
            elif context.size(0) != B:
                raise ValueError(f"Batch mismatch query={B} context={context.size(0)}")

        # Normalize query and context before calculating cosine similarity.
        query_norm = F.normalize(query, dim=-1)
        context_norm = F.normalize(context, dim=-1)

        # Calculate similarity between the query and every context molecule.
        sim = torch.bmm(query_norm, context_norm.transpose(1, 2)).squeeze(1)

        # Select the k most similar context molecules.
        # k cannot be larger than the available context size.
        k = min(self.top_k, context.size(1))
        _, idx = torch.topk(sim, k=k, dim=-1)

        # Gather the selected context representations.
        topk_context = torch.gather(
            context, 1, idx.unsqueeze(-1).expand(-1, -1, D),
        )
        return topk_context

    # -------------------------------------------------
    # Single retrieval step
    # -------------------------------------------------
    def retrieval_step(self, query, sa, si, context):
        # Select only the context molecules most relevant to the query.
        context_topk = self.topk_context(query, context)

        # Project query, active supports, and inactive supports
        # into their respective representations.
        q_proj = self.query_proj(query)
        sa_proj = self.active_proj(sa)
        si_proj = self.inactive_proj(si)

        # Combine query and support representations into one sequence
        # so that they can be processed together by the Hopfield layer.
        s = torch.cat((q_proj, sa_proj, si_proj), dim=1)

        # Retrieve relevant information from the selected context.
        s_h = self.hopfield(
            query=s,
            key=context_topk,
            value=context_topk,
        )

        # Split the retrieved sequence back into query,
        # active-support, and inactive-support representations.
        q_h = s_h[:, 0:1]
        sa_h = s_h[:, 1:1 + sa_proj.shape[1]]
        si_h = s_h[:, 1 + sa_proj.shape[1]:]

        # Convert the learnable gate parameters to values between 0 and 1.
        # This determines how much of the retrieved update is applied.
        q_gate = torch.sigmoid(self.query_gate).view(1, 1, -1)
        s_gate = torch.sigmoid(self.support_gate).view(1, 1, -1)

        # Apply gated residual updates to the original representations.
        query = query + q_gate * (q_h - q_proj)
        sa = sa + s_gate * (sa_h - sa_proj)
        si = si + s_gate * (si_h - si_proj)
        return query, sa, si

    def ffn_block(self, x):
        # Normalize the representation before the FFN,
        # then add the FFN output through a residual connection.
        x_norm = self.ffn_norm(x)
        return x + self.ffn(x_norm)

    # -------------------------------------------------
    # Forward pass
    # -------------------------------------------------
    def forward(self, query, support_actives, support_inactives, context):

        # Pre-normalize all input representations before processing.
        query = self.pre_norm(query)
        support_actives = self.pre_norm(support_actives)
        support_inactives = self.pre_norm(support_inactives)
        context = self.pre_norm(context)

        # Repeat Hopfield retrieval for the configured number of steps.
        for _ in range(self.num_steps):
            query, support_actives, support_inactives = self.retrieval_step(
                query, support_actives, support_inactives, context,
            )

            # Further refine the representations after each
            # Hopfield retrieval step using the FFN.
            query = self.ffn_block(query)
            support_actives = self.ffn_block(support_actives)
            support_inactives = self.ffn_block(support_inactives)

        return query, support_actives, support_inactives
