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
    def __init__(self, n_embd, n_head=2):
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
class MoE(nn.Module):
    def __init__(self, d_model, n_experts=4, hidden_dim=32, dropout=0.05):
        super().__init__()
        self.router = nn.Linear(d_model, n_experts)
        self.n_experts = n_experts

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, hidden_dim * 2),
                SwiGLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
            ) for _ in range(n_experts)
        ])

    def forward(self, x, query=None, soft=True, support_count=None):
        B, T, D = x.shape

        n_experts_use = self.n_experts
        if support_count is not None:
            n_experts_use = min(self.n_experts, max(1, support_count // 2))

        experts = self.experts[:n_experts_use]
        logits = self.router(x)[:, :, :n_experts_use]

        if query is not None:
            q_vec = query.mean(dim=1, keepdim=True)
            sim = torch.einsum("bqd,btd->bqt", q_vec, x).squeeze(1)
            logits = logits + sim.unsqueeze(-1)

        probs = F.softmax(logits, dim=-1)

        out = sum(expert(x) * probs[..., i:i+1]
                  for i, expert in enumerate(experts))

        return out

# -----------------------------
# Cross Attention
# -----------------------------
class ScaledDotProductCrossAttention(nn.Module):
    def __init__(self, config, attn_dropout=0.1, temp=1.0):
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

        q = q * (1.0 / math.sqrt(self.head_dim))
        temp = torch.exp(self.log_temp).clamp(0.1, 10)

        out = torch.zeros_like(q)

        if Na > 0:
            out_a = F.scaled_dot_product_attention(
                q / temp, k[:, :, :Na], v[:, :, :Na]
            )
            out += out_a / math.sqrt(max(Na, 1))

        if Ni > 0:
            out_i = F.scaled_dot_product_attention(
                q / temp, k[:, :, Na:], v[:, :, Na:]
            )
            out -= out_i / math.sqrt(max(Ni, 1))

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
        self.moe = MoE(config.n_embd)

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

        kv = kv + torch.sigmoid(self.gate_ffn) * self.moe(
            self.ffn_norm(kv),
            query=q,
            soft=self.training,
            support_count=kv.size(1)
        )

        return q, kv

# -----------------------------
# CrossAttentionModule
# -----------------------------
class CrossAttentionModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.model_dim = cfg.model.transformer.activity_embedding_dim
        config = GPTConfig(n_embd=self.model_dim)

        # ✅ FIX: no projection needed
        self.input_proj = nn.Identity()

        self.embed = InputEmbedding(self.model_dim)
        self.block = TransformerBlock(config)

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

        q, kv = self.block(query, kv, kv_mask, n_actives=n_actives)

        actives_out = kv[:, :n_actives]
        inactives_out = kv[:, n_actives:]

        return q, actives_out, inactives_out