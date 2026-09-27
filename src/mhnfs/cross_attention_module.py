import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module):
    # Initialize linear layers with Xavier initialization.
    # Biases are initialized to zero.
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
        # Normalize each token representation by its root mean square.
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


class ActivityEncoding(nn.Module):
    """
    Fixed activity encoding for distinguishing query and support molecules.

    The query receives 0, active molecules receive +1, and inactive
    molecules receive -1. The encoding is added directly to the
    representations and is not learned.
    """

    def forward(self, query, actives, inactives):
        # Keep the query unchanged.
        query = query + 0.0  # explicit no-op for clarity, query stays at 0

        # Add a positive constant to active support molecules.
        actives = actives + torch.ones_like(actives)

        # Add a negative constant to inactive support molecules.
        inactives = inactives - torch.ones_like(inactives)

        return query, actives, inactives


class UnifiedSelfAttention(nn.Module):
    """
    Self-attention over the complete query and support sequence.

    The input sequence contains the query, active supports, and inactive
    supports. A single padding mask prevents padded support positions
    from being used as keys during attention.
    """

    def __init__(self, config, attn_dropout=0.1):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.attn_dropout = attn_dropout

        # Project the input representations into queries, keys, and values.
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # Learnable attention temperature controlling the attention scale.
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0)))

    def forward(self, x, padding_mask):
        """
        x: [B, T, D]
           T = query + active supports + inactive supports

        padding_mask: [B, T]
           True = valid molecule, False = padding
        """
        B, T, D = x.shape

        # Create query, key, and value representations.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split the embedding dimension into multiple attention heads.
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scale queries using the learned temperature.
        temp = torch.exp(self.log_temp).clamp(0.1, 10)
        q = q / (math.sqrt(self.head_dim) * temp)

        # Create one mask for the complete sequence.
        # Padded molecules receive -inf so they cannot contribute to attention.
        key_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        attn_bias = torch.zeros(B, 1, 1, T, device=x.device)
        attn_bias = attn_bias.masked_fill(~key_mask, float('-inf'))

        # Apply self-attention over the complete query/support sequence.
        # Dropout is only active during training.
        #out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.attn_dropout if self.training else 0.0
        )

        # Merge the attention heads back into the original embedding dimension.
        out = out.transpose(1, 2).contiguous().reshape(B, T, D)

        # Project the combined representation back to the model dimension.
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer-style block containing:
    - RMS normalization
    - unified self-attention
    - gated residual connection
    - feed-forward network
    - gated FFN residual connection
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

        # Start with a small residual contribution from both attention and FFN.
        self.gate_attn = nn.Parameter(torch.tensor(-4.0))  # sigmoid ≈ 0.018
        self.gate_ffn = nn.Parameter(torch.tensor(-4.0))   # sigmoid ≈ 0.018

    def forward(self, x, padding_mask):
        # Normalize the input before self-attention.
        delta = self.attn(self.x_norm(x), padding_mask)

        # Normalize the attention output before applying the residual update.
        delta = self.delta_norm(delta)

        # Add the attention update through a learnable residual gate.
        x = x + torch.sigmoid(self.gate_attn) * delta

        # Apply the FFN and add its output through a separate residual gate.
        x = x + torch.sigmoid(self.gate_ffn) * self.ffn(self.ffn_norm(x))

        return x


class CrossAttentionModule(nn.Module):
    """
    Transformer-style module for interaction between the query and support set.

    The module:
    - uses unified self-attention over query and support molecules
    - adds fixed activity information to distinguish active and inactive supports
    - uses a module-level residual gate
    - optionally applies stochastic depth to later Transformer blocks
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

        # Add fixed activity information to the query and support representations.
        self.activity_encoding = ActivityEncoding()

        # Stack multiple Transformer blocks for query-support interaction.
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])

        # Controls the overall contribution of the Cross-Attention Module.
        self.module_gate = nn.Parameter(torch.tensor(-4.0))

        self.apply(init_weights)

    def forward(self, query, actives, inactives, act_mask, inact_mask):
        B = query.size(0)

        # Keep the original representations for the final module-level residual.
        query_in, actives_in, inactives_in = query, actives, inactives

        # Add fixed information indicating whether each support molecule
        # is active or inactive.
        query, actives, inactives = self.activity_encoding(query, actives, inactives)

        n_actives = actives.size(1)
        n_inactives = inactives.size(1)

        # Combine query and both support groups into one sequence.
        x = torch.cat([query, actives, inactives], dim=1)

        # The query is always valid, while support validity is given by
        # the active and inactive masks.
        query_mask = torch.ones(B, 1, dtype=torch.bool, device=query.device)
        padding_mask = torch.cat([query_mask, act_mask, inact_mask], dim=1)

        # Process the complete sequence through the Transformer blocks.
        for i, block in enumerate(self.blocks):
            # During training, later blocks can be randomly skipped
            # as a form of stochastic-depth regularization.
            if self.training and i > 0 and torch.rand(1).item() < self.stochastic_depth_prob:
                continue  # skip this block this forward pass (stochastic depth)
            x = block(x, padding_mask)

        # Split the sequence back into query, active, and inactive representations.
        query_out = x[:, 0:1, :]
        actives_out = x[:, 1:1 + n_actives, :]
        inactives_out = x[:, 1 + n_actives:1 + n_actives + n_inactives, :]

        # Apply the complete module update through a learnable residual gate.
        gate = torch.sigmoid(self.module_gate)
        query_out = query_in + gate * (query_out - query_in)
        actives_out = actives_in + gate * (actives_out - actives_in)
        inactives_out = inactives_in + gate * (inactives_out - inactives_in)

        return query_out, actives_out, inactives_out
