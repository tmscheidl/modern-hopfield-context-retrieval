import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MyHopfield(nn.Module):
    """
    Modern Hopfield Network layer used for associative retrieval.

    The layer supports multi-head retrieval, learnable and head-specific
    temperature parameters, optional input normalization and projections,
    residual updates, masking, dropout, and iterative retrieval.
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

        # Each attention head operates on an equal-sized part of the
        # input representation.
        assert input_size % num_heads == 0, "input_size must be divisible by num_heads"

        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        self.use_layer_norm = use_layer_norm
        self.use_projection = use_projection
        self.normalize_patterns = normalize_patterns

        self.beta_min = beta_min
        self.beta_max = beta_max

        # Dropout applied to the association weights and to the retrieved
        # representation before adding it to the residual connection.
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

        # -----------------------------
        # Projections
        # -----------------------------
        if use_projection:
            # Project the input into separate query, key, and value
            # representations before associative retrieval.
            self.q_proj = nn.Linear(input_size, input_size)
            self.k_proj = nn.Linear(input_size, input_size)
            self.v_proj = nn.Linear(input_size, input_size)
            self.out_proj = nn.Linear(input_size, input_size)

        # -----------------------------
        # Learnable temperature
        # -----------------------------
        # Each head has its own learnable beta parameter. Beta controls
        # the sharpness of the association distribution.
        self.beta_param = nn.Parameter(torch.ones(num_heads) * init_beta)

        # -----------------------------
        # LayerNorm
        # -----------------------------
        if use_layer_norm:
            # Normalize query, key, and value representations independently
            # before the associative retrieval operation.
            self.norm_query = nn.LayerNorm(input_size)
            self.norm_key = nn.LayerNorm(input_size)
            self.norm_value = nn.LayerNorm(input_size)

        self.reset_parameters()

    # -------------------------------------------------
    # Positive beta via softplus
    # -------------------------------------------------
    @property
    def beta(self):
        # Softplus guarantees that the temperature parameter remains positive.
        return F.softplus(self.beta_param)

    # -------------------------------------------------
    # Head utilities
    # -------------------------------------------------
    def split_heads(self, x):
        B, T, D = x.shape

        # Reshape the representation into multiple independent heads.
        x = x.view(B, T, self.num_heads, self.head_dim)

        # Move the head dimension before the sequence dimension to obtain
        # the format [batch, heads, sequence, head dimension].
        return x.transpose(1, 2)  # [B, H, T, Dh]

    def merge_heads(self, x):
        B, H, T, Dh = x.shape

        # Move the sequence dimension back before combining all heads
        # into the original representation dimension.
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, H * Dh)

    # -------------------------------------------------
    # Temperature clamp
    # -------------------------------------------------
    def _get_beta(self):
        # Limit beta to a predefined range to avoid excessively sharp
        # or flat association distributions.
        beta = torch.clamp(self.beta, self.beta_min, self.beta_max)

        # Expand a scalar beta if only one temperature value is available.
        if beta.numel() == 1:
            beta = beta.expand(self.num_heads)

        # Reshape beta so that a separate temperature can be applied
        # to each attention head.
        return beta.view(1, self.num_heads, 1, 1)

    # -------------------------------------------------
    # Mask support
    # -------------------------------------------------
    def _apply_mask(self, scores, mask):
        if mask is None:
            return scores

        # Convert lower-dimensional masks to a shape compatible with
        # the multi-head association score tensor.
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(1)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Invalid positions receive -infinity so that their association
        # probability becomes zero after the softmax operation.
        mask = mask.bool()
        return scores.masked_fill(~mask, float("-inf"))

    # -------------------------------------------------
    # Core Hopfield forward
    # -------------------------------------------------
    def forward(self, query, key=None, value=None, mask=None):

        # Keep the original query for the residual connection.
        residual = query

        # If no separate key/value representations are provided,
        # perform associative retrieval from the query itself.
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
            # Transform query, key, and value into their retrieval
            # representations.
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        else:
            q, k, v = query, key, value

        # Normalize patterns (cosine similarity)
        if self.normalize_patterns:
            # L2 normalization changes the dot product into a cosine-like
            # similarity measure and reduces the influence of vector magnitude.
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)

        # Multi-head
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Attention scores

        # Calculate pairwise query-key similarities for associative retrieval.
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply the head-specific temperature to control the sharpness
        # of the retrieval distribution.
        beta = self._get_beta()
        scores = scores * beta

        # Remove invalid or padded positions from the retrieval operation.
        scores = self._apply_mask(scores, mask)

        # Stability tricks
        # Limit extreme values and subtract the maximum score before softmax
        # to improve numerical stability.
        scores = torch.clamp(scores, -50, 50)
        scores = scores - scores.max(dim=-1, keepdim=True)[0]

        # Convert similarity scores into association weights.
        attn = F.softmax(scores, dim=-1)

        # Replace possible numerical NaN values with finite values.
        attn = torch.nan_to_num(attn)

        # Regularize the association weights during training.
        attn = self.attn_dropout(attn)

        # Retrieve a weighted combination of the value patterns.
        out = torch.matmul(attn, v)

        # Merge the independently processed heads back into one representation.
        out = self.merge_heads(out)

        if self.use_projection:
            # Project the combined representation back into the original
            # input space.
            out = self.out_proj(out)

        # Apply dropout before the residual update.
        out = self.residual_dropout(out)

        # Residual update: retain the original query and add the
        # retrieved contextual information.
        return residual + out

    # -------------------------------------------------
    # Energy function
    # -------------------------------------------------
    def compute_energy(self, query, key=None):

        # If no separate patterns are provided, use the query itself
        # as the retrieval patterns.
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
            # Use normalized representations for the same similarity
            # calculation as in the main retrieval operation.
            q = F.normalize(q, dim=-1, eps=1e-6)
            k = F.normalize(k, dim=-1, eps=1e-6)

        q = self.split_heads(q)
        k = self.split_heads(k)

        # Calculate query-key association scores.
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply the learned temperature for each head.
        beta = self._get_beta()
        scores = scores * beta

        # Compute the log-sum-exp term used to obtain the
        # Hopfield energy.
        lse = torch.logsumexp(scores, dim=-1)

        # Avoid division by zero when using the temperature in the
        # final energy calculation.
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
        # Start the iterative retrieval process from the original query.
        state = query
        prev_energy = None
        energies = []

        for _ in range(max_steps):
            # Perform another Hopfield retrieval/update step.
            state = self.forward(state, key, value, mask)

            # Measure the energy of the updated state.
            energy = self.compute_energy(state, key).mean()

            if return_energy:
                energies.append(energy.item())

            # Stop early when the energy has converged and the change
            # between consecutive iterations is below the tolerance.
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

        # Use the query itself as the retrieval patterns when no separate
        # key representation is provided.
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

        # Split query and key representations into independent heads.
        q = self.split_heads(q)
        k = self.split_heads(k)

        # Calculate pairwise association scores.
        scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply the head-specific temperature.
        beta = self._get_beta()
        scores = scores * beta

        # Exclude masked positions from the association matrix.
        scores = self._apply_mask(scores, mask)

        # Apply the same numerical stability operations as in the
        # main retrieval operation.
        scores = torch.clamp(scores, -50, 50)
        scores = scores - scores.max(dim=-1, keepdim=True)[0]

        # Convert scores into normalized association probabilities.
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)

        return attn

    # -------------------------------------------------
    # Initialization
    # -------------------------------------------------
    def reset_parameters(self):

        if self.use_projection:
            # Xavier initialization provides a suitable starting scale
            # for the projection matrices.
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.out_proj.weight)

            # Start all projection biases at zero.
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

        if self.use_layer_norm:
            # Initialize LayerNorm to initially preserve the normalized
            # representation without adding a learned shift.
            nn.init.ones_(self.norm_query.weight)
            nn.init.zeros_(self.norm_query.bias)

            nn.init.ones_(self.norm_key.weight)
            nn.init.zeros_(self.norm_key.bias)

            nn.init.ones_(self.norm_value.weight)
            nn.init.zeros_(self.norm_value.bias)

# For the improvement
        # Optional Multi-Head Hopfield x
        # β Head-Specific x
        # Dropout on Association Matrix x
        # Residual Connection x
        # Mask Support x
        # Temperature Clamping x
        # Energy Function x
        # Iterative Retrieval x
        # Initialization Improvements x
        # Numerical Stability x
