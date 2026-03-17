import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

# Import your Hopfield module relative to the repo root
from hopfield.my_hopfield import MyHopfield

# -------------------------------------------------
# Weight initialization
# -------------------------------------------------
def init_weights(module_type, module):
    if module_type == "linear" and isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# -------------------------------------------------
# ContextModule
# -------------------------------------------------
class ContextModule(nn.Module):
    """
    Stable Context Module

    Features
    --------
    • Multi-step Hopfield retrieval
    • Top-k memory selection
    • Transformer-style FFN refinement
    • Gated residual updates
    """

    def __init__(self, cfg, top_k=32):
        super().__init__()

        dim = cfg.model.associationSpace_dim
        ffn_mult = 4

        self.num_steps = cfg.model.hopfield.num_steps
        self.top_k = top_k

        # -------------------------------------------------
        # Hopfield Memory
        # -------------------------------------------------
        self.hopfield = MyHopfield(
            input_size=dim,
            num_heads=cfg.model.hopfield.heads,
            init_beta=cfg.model.hopfield.beta,
            attn_dropout=cfg.model.hopfield.dropout,
        )

        self.hopfield.apply(partial(init_weights, "linear"))

        # -------------------------------------------------
        # Projections
        # -------------------------------------------------
        self.query_proj = nn.Linear(dim, dim)
        self.active_proj = nn.Linear(dim, dim)
        self.inactive_proj = nn.Linear(dim, dim)

        # -------------------------------------------------
        # Gates
        # -------------------------------------------------
        self.query_gate = nn.Parameter(torch.full((dim,), -0.5))
        self.support_gate = nn.Parameter(torch.full((dim,), -1.0))

        # -------------------------------------------------
        # Normalization
        # -------------------------------------------------
        self.pre_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

        # -------------------------------------------------
        # Feed Forward Network
        # -------------------------------------------------
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    # -------------------------------------------------
    # L2 normalization
    # -------------------------------------------------
    def l2_norm(self, x):
        return F.normalize(x, dim=-1, eps=1e-8)

    # -------------------------------------------------
    # Top-K context selection
    # -------------------------------------------------
    def topk_context(self, query, context):

        """
        query:   [B,1,D]
        context: [Nc,D] or [B,Nc,D]
        return:  [B,top_k,D]
        """

        B = query.size(0)
        D = query.size(-1)

        if context.dim() == 2:
            context = context.unsqueeze(0).expand(B, -1, -1)

        elif context.dim() == 3:
            if context.size(0) == 1 and B > 1:
                context = context.expand(B, -1, -1)

            elif context.size(0) != B:
                raise ValueError(
                    f"Batch mismatch query={B} context={context.size(0)}"
                )

        query_norm = F.normalize(query, dim=-1)
        context_norm = F.normalize(context, dim=-1)

        sim = torch.bmm(
            query_norm,
            context_norm.transpose(1, 2)
        ).squeeze(1)

        k = min(self.top_k, context.size(1))

        _, idx = torch.topk(sim, k=k, dim=-1)

        topk_context = torch.gather(
            context,
            1,
            idx.unsqueeze(-1).expand(-1, -1, D),
        )

        return topk_context

    # -------------------------------------------------
    # Single retrieval step
    # -------------------------------------------------
    def retrieval_step(self, query, sa, si, context):

        # select top-k memory
        context_topk = self.topk_context(query, context)

        # projections
        q_proj = self.query_proj(query)
        sa_proj = self.active_proj(sa)
        si_proj = self.inactive_proj(si)

        # combine states
        s = torch.cat((q_proj, sa_proj, si_proj), dim=1)

        # Hopfield retrieval
        s_h = self.hopfield(
            query=s,
            key=context_topk,
            value=context_topk,
        )

        # split states
        q_h = s_h[:, 0:1]
        sa_h = s_h[:, 1:1 + sa_proj.shape[1]]
        si_h = s_h[:, 1 + sa_proj.shape[1]:]

        # gates
        q_gate = torch.sigmoid(self.query_gate).view(1, 1, -1)
        s_gate = torch.sigmoid(self.support_gate).view(1, 1, -1)

        # residual updates
        query = query + q_gate * (q_h - q_proj)
        sa = sa + s_gate * (sa_h - sa_proj)
        si = si + s_gate * (si_h - si_proj)

        return query, sa, si

    # -------------------------------------------------
    # Transformer FFN block
    # -------------------------------------------------
    def ffn_block(self, x):

        x_norm = self.ffn_norm(x)

        return x + self.ffn(x_norm)

    # -------------------------------------------------
    # Forward pass
    # -------------------------------------------------
    def forward(
        self,
        query,
        support_actives,
        support_inactives,
        context
    ):

        # PreNorm
        query = self.pre_norm(query)
        support_actives = self.pre_norm(support_actives)
        support_inactives = self.pre_norm(support_inactives)
        context = self.pre_norm(context)

        # multi-step retrieval
        for _ in range(self.num_steps):

            query, support_actives, support_inactives = self.retrieval_step(
                query,
                support_actives,
                support_inactives,
                context,
            )

            # FFN refinement
            query = self.ffn_block(query)
            support_actives = self.ffn_block(support_actives)
            support_inactives = self.ffn_block(support_inactives)

        # final normalization
        query = self.l2_norm(query)
        support_actives = self.l2_norm(support_actives)
        support_inactives = self.l2_norm(support_inactives)

        return query, support_actives, support_inactives
