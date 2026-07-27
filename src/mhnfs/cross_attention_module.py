import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class GPTConfig:
    def __init__(self, n_embd, n_head=8):
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        assert self.head_dim * n_head == n_embd


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class ActivityEncoding(nn.Module):
    """
    V24 change (from professor's design): additive CONSTANT encoding
    instead of a learned type embedding. query gets 0, actives get +1,
    inactives get -1, broadcast across the embedding dimension.
    This is fixed, not learned - simpler, and a genuinely different
    mechanism than the InputEmbedding type-embedding used previously.
    """

    def forward(self, query, actives, inactives):
        query = query + 0.0  # explicit no-op for clarity, query stays at 0
        actives = actives + torch.ones_like(actives)
        inactives = inactives - torch.ones_like(inactives)
        return query, actives, inactives


class UnifiedSelfAttention(nn.Module):
    """
    V24 change (from professor's design): ONE self-attention over the
    concatenated [query, actives, inactives] sequence, governed by a
    single padding mask. Replaces the previous split active/inactive
    cross-attention with separate scaling paths. Query can now also
    attend to itself and the full set, not just support molecules.
    """

    def __init__(self, config, attn_dropout=0.1):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, x, padding_mask):
        """
        x: [B, T, D]  (T = 1 + Na + Ni, query + actives + inactives)
        padding_mask: [B, T] bool, True = valid (real) molecule
        """
        B, T, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        temp = torch.exp(self.log_temp).clamp(0.1, 10)
        q = q / (math.sqrt(self.head_dim) * temp)

        # single unified mask: [B, 1, 1, T] -> broadcasts over query positions too
        key_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        attn_bias = torch.zeros(B, 1, 1, T, device=x.device)
        attn_bias = attn_bias.masked_fill(~key_mask, float('-inf'))

        #out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.attn_dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    Kept the same overall shape (RMSNorm, gated residual, FFN) -
    only the attention mechanism inside changed (UnifiedSelfAttention
    instead of the previous split active/inactive cross-attention).
    """

    def __init__(self, config):
        super().__init__()
        self.x_norm = RMSNorm(config.n_embd)
        self.attn = UnifiedSelfAttention(config)
        self.delta_norm = RMSNorm(config.n_embd)

        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.Dropout(0.5),
        )

        self.gate_attn = nn.Parameter(torch.tensor(-4.0))  # sigmoid ≈ 0.018
        self.gate_ffn = nn.Parameter(torch.tensor(-4.0))   # sigmoid ≈ 0.018

    def forward(self, x, padding_mask):
        delta = self.attn(self.x_norm(x), padding_mask)
        delta = self.delta_norm(delta)
        x = x + torch.sigmoid(self.gate_attn) * delta
        x = x + torch.sigmoid(self.gate_ffn) * self.ffn(self.ffn_norm(x))
        return x


class CrossAttentionModule(nn.Module):
    """
    V24 changes:
    1. Unified self-attention (professor-style) instead of split active/
       inactive cross-attention - kept inside the existing transformer
       block shape (RMSNorm, gating, FFN unchanged in spirit).
    2. Additive constant activity encoding (professor-style) instead of
       a learned type embedding.
    3. Module-level residual gate: the whole cross-attention module's
       effect can be downweighted by a learnable gate, giving the model
       an "escape hatch" if cross-attention isn't helping for a given
       batch.
    4. Stochastic depth: when stacking multiple blocks, randomly skip
       later blocks during training (never at eval) as a regularizer.
    """

    def __init__(self, cfg):
        super().__init__()
        self.model_dim = cfg.model.associationSpace_dim
        num_heads = getattr(cfg.model.transformer, "number_heads", 8)
        num_layers = getattr(cfg.model.transformer, "num_layers", 2)
        self.stochastic_depth_prob = getattr(
            cfg.model.transformer, "stochastic_depth_prob", 0.1
        )

        config = GPTConfig(n_embd=self.model_dim, n_head=num_heads)

        self.activity_encoding = ActivityEncoding()
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])

        # module-level residual gate (point 3 above)
        self.module_gate = nn.Parameter(torch.tensor(-4.0))

        self.apply(init_weights)

    def forward(self, query, actives, inactives, act_mask, inact_mask):
        B = query.size(0)

        # save originals for the module-level residual
        query_in, actives_in, inactives_in = query, actives, inactives

        # additive constant activity encoding (point 2 above)
        query, actives, inactives = self.activity_encoding(query, actives, inactives)

        n_actives = actives.size(1)
        n_inactives = inactives.size(1)

        x = torch.cat([query, actives, inactives], dim=1)

        query_mask = torch.ones(B, 1, dtype=torch.bool, device=query.device)
        padding_mask = torch.cat([query_mask, act_mask, inact_mask], dim=1)

        for i, block in enumerate(self.blocks):
            if self.training and i > 0 and torch.rand(1).item() < self.stochastic_depth_prob:
                continue  # skip this block this forward pass (stochastic depth)
            x = block(x, padding_mask)

        query_out = x[:, 0:1, :]
        actives_out = x[:, 1:1 + n_actives, :]
        inactives_out = x[:, 1 + n_actives:1 + n_actives + n_inactives, :]

        # module-level residual gate (point 3 above)
        gate = torch.sigmoid(self.module_gate)
        query_out = query_in + gate * (query_out - query_in)
        actives_out = actives_in + gate * (actives_out - actives_in)
        inactives_out = inactives_in + gate * (inactives_out - inactives_in)

        return query_out, actives_out, inactives_out
