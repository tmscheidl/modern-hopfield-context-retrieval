import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utilities
# -----------------------------
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# -----------------------------
# GPT-like config
# -----------------------------
class GPTConfig:
    def __init__(self, n_embd, n_head=8):
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        assert self.head_dim * n_head == n_embd

# -----------------------------
# RMSNorm
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight

# -----------------------------
# Input embedding
# -----------------------------
class InputEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.type_emb = nn.Embedding(3, dim)

    def forward(self, x, types):
        return x + self.type_emb(types)

# -----------------------------
# SwiGLU
# -----------------------------
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)

# -----------------------------
# Mixture-of-Experts
# -----------------------------


# -----------------------------
# Cross Attention
# -----------------------------
class ScaledDotProductCrossAttention(nn.Module):
    def __init__(self, config, attn_dropout=0.5, temp=1.0):
        super().__init__()

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.attn_dropout = attn_dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.active_bias = nn.Embedding(1, config.n_embd)
        self.inactive_bias = nn.Embedding(1, config.n_embd)

        self.log_temp = nn.Parameter(torch.log(torch.tensor(temp)))

    def forward(self, q, kv, kv_mask=None, n_actives=None):
        B, Tq, D = q.shape
        Tk = kv.size(1)
        Na = int(n_actives)
        Ni = Tk - Na

        q = self.q_proj(q)
        kv = kv.clone()
        kv[:, :Na] += self.active_bias.weight
        kv[:, Na:] += self.inactive_bias.weight
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = q.view(B, Tq, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, Tk, self.n_head, self.head_dim).transpose(1, 2)

        temp = torch.exp(self.log_temp).clamp(0.1, 10)
        q = q / (math.sqrt(self.head_dim) * temp)

        out = torch.zeros_like(q)

        if Na > 0 and kv_mask is not None:
            # mask for active positions only: [B, Na] → [B, 1, 1, Na]
            act_mask = kv_mask[:, :Na].unsqueeze(1).unsqueeze(2)  # True=valid
            attn_bias_a = torch.zeros(B, 1, 1, Na, device=q.device)
            attn_bias_a = attn_bias_a.masked_fill(~act_mask, float('-inf'))
            out_a = F.scaled_dot_product_attention(q, k[:, :, :Na], v[:, :, :Na],
                                                    attn_mask=attn_bias_a)
            valid_a = act_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
            out += out_a / torch.sqrt(valid_a)

        if Ni > 0 and kv_mask is not None:
            # mask for inactive positions only: [B, Ni] → [B, 1, 1, Ni]
            inact_mask = kv_mask[:, Na:].unsqueeze(1).unsqueeze(2)  # True=valid
            attn_bias_i = torch.zeros(B, 1, 1, Ni, device=q.device)
            attn_bias_i = attn_bias_i.masked_fill(~inact_mask, float('-inf'))
            out_i = F.scaled_dot_product_attention(q, k[:, :, Na:], v[:, :, Na:],
                                                    attn_mask=attn_bias_i)
            valid_i = inact_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
            out -= out_i / torch.sqrt(valid_i)

        out = out.transpose(1, 2).contiguous().reshape(B, Tq, D)
        return self.out_proj(out)

# -----------------------------
# Transformer block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.q_norm = RMSNorm(config.n_embd)
        self.kv_norm_a = RMSNorm(config.n_embd)
        self.kv_norm_i = RMSNorm(config.n_embd)

        self.cross_attn = ScaledDotProductCrossAttention(config)
        self.delta_norm = RMSNorm(config.n_embd)

        self.ffn_norm = RMSNorm(config.n_embd)
        #self.moe = MoE(config.n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 2),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(config.n_embd * 2, config.n_embd),
            nn.Dropout(0.5),
        )

        self.gate_attn = nn.Parameter(torch.tensor(0.3))
        self.gate_ffn = nn.Parameter(torch.tensor(0.7))

    def forward(self, q, kv, kv_mask, n_actives):
        Na = n_actives

        kv_a = self.kv_norm_a(kv[:, :Na])
        kv_i = self.kv_norm_i(kv[:, Na:])
        kv_combined = torch.cat([kv_a, kv_i], dim=1)

        delta = self.cross_attn(self.q_norm(q), kv_combined, kv_mask, n_actives=Na)
        delta = self.delta_norm(delta)

        q = q + torch.sigmoid(self.gate_attn) * delta

        kv = kv + torch.sigmoid(self.gate_ffn) * self.ffn(self.ffn_norm(kv))

        return q, kv

# -----------------------------
# CrossAttentionModule
# -----------------------------
class CrossAttentionModule(nn.Module):
    def __init__(self, cfg, num_layers=2):
        super().__init__()
        self.model_dim = cfg.model.transformer.activity_embedding_dim
        config = GPTConfig(n_embd=self.model_dim, n_head=8)
        self.input_proj = nn.Identity()
        self.embed = InputEmbedding(self.model_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(num_layers)])
        self.apply(init_weights)

    def forward(self, query, actives, inactives, act_mask, inact_mask):
        B = query.size(0)
        query = self.input_proj(query)
        actives = self.input_proj(actives)
        inactives = self.input_proj(inactives)
        n_actives = actives.size(1)
        kv = torch.cat([actives, inactives], dim=1)
        kv_mask = torch.cat([act_mask, inact_mask], dim=1)

        q_types = torch.zeros(B, query.size(1), dtype=torch.long, device=query.device)
        kv_types = torch.cat([
            torch.ones(B, actives.size(1), device=query.device),
            torch.full((B, inactives.size(1)), 2, device=query.device)
        ], dim=1).long()

        query = self.embed(query, q_types)
        kv = self.embed(kv, kv_types)

        for block in self.blocks:
            query, kv = block(query, kv, kv_mask, n_actives=n_actives)

        actives_out = kv[:, :n_actives]
        inactives_out = kv[:, n_actives:]
        return query, actives_out, inactives_out