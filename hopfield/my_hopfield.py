import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyHopfield(nn.Module):
    """
    MyHopfield

    A modern Hopfield Network layer with:
    - Multi-head associative retrieval
    - Learnable temperature (beta)
    - Residual updates
    - Mask support
    - Iterative retrieval
    - Energy computation
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int = 4,
        use_layer_norm: bool = True,
        use_projection: bool = True,
        init_beta: float = 1.0,
        attn_dropout: float = 0.1,
        residual_dropout: float = 0.1,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        normalize_patterns: bool = True,
    ):
        super().__init__()

        assert input_size % num_heads == 0, "input_size must be divisible by num_heads"

        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        self.use_layer_norm = use_layer_norm
        self.use_projection = use_projection
        self.normalize_patterns = normalize_patterns

        self.beta_min = beta_min
        self.beta_max = beta_max

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

        # -----------------------------
        # Projections
        # -----------------------------
        if use_projection:
            self.q_proj = nn.Linear(input_size, input_size)
            self.k_proj = nn.Linear(input_size, input_size)
            self.v_proj = nn.Linear(input_size, input_size)
            self.out_proj = nn.Linear(input_size, input_size)

        # -----------------------------
        # Learnable temperature
        # -----------------------------
        self.beta_param = nn.Parameter(torch.ones(num_heads) * init_beta)

        # -----------------------------
        # LayerNorm
        # -----------------------------
        if use_layer_norm:
            self.norm_query = nn.LayerNorm(input_size)
            self.norm_key = nn.LayerNorm(input_size)
            self.norm_value = nn.LayerNorm(input_size)

        self.reset_parameters()

    # -------------------------------------------------
    # Positive beta via softplus
    # -------------------------------------------------
    @property
    def beta(self):
        return F.softplus(self.beta_param)

    # -------------------------------------------------
    # Head utilities
    # -------------------------------------------------
    def split_heads(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, T, Dh]

    def merge_heads(self, x):
        B, H, T, Dh = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * Dh)

    # -------------------------------------------------
    # Temperature clamp
    # -------------------------------------------------
    def _get_beta(self):
        beta = torch.clamp(self.beta, self.beta_min, self.beta_max)

        if beta.numel() == 1:
            beta = beta.expand(self.num_heads)

        return beta.view(1, self.num_heads, 1, 1)

    # -------------------------------------------------
    # Mask support
    # -------------------------------------------------
    def _apply_mask(self, scores, mask):
        if mask is None:
            return scores

        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        mask = mask.bool()
        return scores.masked_fill(~mask, float("-inf"))

    # -------------------------------------------------
    # Core Hopfield forward
    # -------------------------------------------------
    def forward(self, query, key=None, value=None, mask=None):

        residual = query

        if key is None:
            key = query
        if value is None:
            value = key

        # LayerNorm
        if self.use_layer_norm:
            query = self.norm_query(query)
            key = self.norm_key(key)
            value = self.norm_value(value)

        # Projections
        if self.use_projection:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        else:
            q, k, v = query, key, value

        # Normalize patterns (cosine similarity)
        if self.normalize_patterns:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)

        # Multi-head
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))

        beta = self._get_beta()
        scores = scores * beta / math.sqrt(self.head_dim)

        scores = self._apply_mask(scores, mask)

        # Stability tricks
        scores = torch.clamp(scores, -50, 50)
        scores = scores - scores.max(dim=-1, keepdim=True)[0]

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)

        # Merge heads
        out = self.merge_heads(out)

        if self.use_projection:
            out = self.out_proj(out)

        out = self.residual_dropout(out)

        return residual + out

    # -------------------------------------------------
    # Energy function
    # -------------------------------------------------
    def compute_energy(self, query, key=None):

        if key is None:
            key = query

        if self.use_layer_norm:
            query = self.norm_query(query)
            key = self.norm_key(key)

        if self.use_projection:
            q = self.q_proj(query)
            k = self.k_proj(key)
        else:
            q, k = query, key

        if self.normalize_patterns:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)

        q = self.split_heads(q)
        k = self.split_heads(k)

        scores = torch.matmul(q, k.transpose(-2, -1))

        beta = self._get_beta()
        scores = scores * beta / math.sqrt(self.head_dim)

        lse = torch.logsumexp(scores, dim=-1)
        beta_safe = beta.squeeze(-1).clamp(min=1e-6)

        return -lse / beta_safe

    # -------------------------------------------------
    # Iterative retrieval
    # -------------------------------------------------
    def forward_iterative(
        self,
        query,
        key=None,
        value=None,
        mask=None,
        max_steps=10,
        energy_tol=1e-4,
        return_energy=False,
    ):
        state = query
        prev_energy = None
        energies = []

        for _ in range(max_steps):
            state = self.forward(state, key, value, mask)
            energy = self.compute_energy(state, key).mean()

            if return_energy:
                energies.append(energy.item())

            if prev_energy is not None:
                if torch.abs(prev_energy - energy) < energy_tol:
                    break

            prev_energy = energy

        if return_energy:
            return state, energies

        return state

    # -------------------------------------------------
    # Association matrix
    # -------------------------------------------------
    def get_association_matrix(self, query, key=None, mask=None):

        if key is None:
            key = query

        if self.use_layer_norm:
            query = self.norm_query(query)
            key = self.norm_key(key)

        if self.use_projection:
            q = self.q_proj(query)
            k = self.k_proj(key)
        else:
            q, k = query, key

        if self.normalize_patterns:
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)

        q = self.split_heads(q)
        k = self.split_heads(k)

        scores = torch.matmul(q, k.transpose(-2, -1))

        beta = self._get_beta()
        scores = scores * beta / math.sqrt(self.head_dim)

        scores = self._apply_mask(scores, mask)

        scores = torch.clamp(scores, -50, 50)
        scores = scores - scores.max(dim=-1, keepdim=True)[0]

        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)

        return attn

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------
    def reset_parameters(self):

        if self.use_projection:
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

        if self.use_layer_norm:
            nn.init.ones_(self.norm_query.weight)
            nn.init.zeros_(self.norm_query.bias)

            nn.init.ones_(self.norm_key.weight)
            nn.init.zeros_(self.norm_key.bias)

            nn.init.ones_(self.norm_value.weight)
            nn.init.zeros_(self.norm_value.bias)

        beta_init = torch.empty(self.num_heads).uniform_(0.5, 1.5)
        beta_param = torch.log(torch.exp(beta_init) - 1)

        with torch.no_grad():
            self.beta_param.copy_(beta_param)