from math import sqrt
from re import S
import torch.nn as nn
import torch


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        nn.init.trunc_normal_(
            w,
            0,
            sqrt(2 / (in_features + out_features)),
            -3 * sqrt(2 / (in_features + out_features)),
            3 * sqrt(2 / (in_features + out_features)),
        )
        self.weight = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function implements the forward pass of a linear layer.
        x is of shape (N, in_features)
        self.weight is of shape (out_features, in_features)
        The output is of shape (N, out_features)
        """
        return x @ self.weight.t()

    def set_weights(self, weights: torch.Tensor):
        """
        This function sets the weights of the linear layer.
        weights is of shape (out_features, in_features)
        """
        assert weights.shape == (self.out_features, self.in_features)
        self.weight = torch.nn.Parameter(weights)


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Inputs:
            num_embeddings: int Size of the vocabulary, i.e., the number of unique tokens.
            embedding_dim: int Dimensionality of the embedding vectors.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        w = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        nn.init.trunc_normal_(
            w,
            0,
            1,
            -3,
            3,
        )
        self.weight = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function implements the forward pass of an embedding layer.
        x is of shape (batch_size, sequence_length) where each value is an integer in [0, num_embeddings)
        The output is of shape (batch_size, sequence_length, embedding_dim)
        """
        batch_size, sequence_length = x.shape
        result = torch.empty((batch_size, sequence_length, self.embedding_dim))
        for i in range(batch_size):
            for j in range(sequence_length):
                result[i, j] = self.weight[x[i, j]]

        return result

    def set_weights(self, weights: torch.Tensor):
        """
        This function sets the weights of the linear layer.
        weights is of shape (out_features, in_features)
        """
        assert weights.shape == (
            self.num_embeddings,
            self.embedding_dim,
        )
        self.weight = torch.nn.Parameter(weights)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Inputs:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # weights is of shape (d_model,) initialized to all ones
        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
        """
        assert input.shape[-1] == self.d_model
        original_dtype = input.dtype
        # first we transform input to float32 for numerical stability
        input = input.to(torch.float32)
        # Calculate RMS (Root Mean Square)
        # keepdim=True is to make sure broadcasting works well
        rms = torch.sqrt(torch.mean(input**2, dim=-1, keepdim=True) + self.eps)

        # Normalize and apply learnable weights
        normalized = input / rms
        # in the background broadcast self.weights to match the shape of normalized
        result = normalized * self.weights  # Element-wise multiplication

        # Return the result in the original dtype
        return result.to(original_dtype)

    def set_weights(self, weights: torch.Tensor):
        """
        This function sets the weights of the RMSNorm layer.
        weights is of shape (d_model,)
        """
        assert weights.shape == (self.d_model,)
        self.weights = torch.nn.Parameter(weights)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else d_model * 8 // 3
        self.w1 = nn.Parameter(
            torch.randn(self.d_ff, d_model, device=device, dtype=dtype)
        )
        self.w3 = nn.Parameter(
            torch.randn(self.d_ff, d_model, device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            torch.randn(d_model, self.d_ff, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_model

        def SiLU(x):
            return x * torch.sigmoid(x)

        # x @ w1.T gives shape (..., d_ff)
        # x @ w3.T gives shape (..., d_ff)
        # Element-wise multiply and apply SiLU, then multiply by w2.T
        return (SiLU(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T

    def set_weights(
        self, w1: torch.Tensor = None, w2: torch.Tensor = None, w3: torch.Tensor = None
    ):
        """
        This function sets the weights of the SwiGLU layer.
        w1 is of shape (d_ff, d_model)
        w2 is of shape (d_model, d_ff)
        w3 is of shape (d_ff, d_model)
        """
        if w1 is not None:
            assert w1.shape == (self.d_ff, self.d_model)
            self.w1 = torch.nn.Parameter(w1)

        if w2 is not None:
            assert w2.shape == (self.d_model, self.d_ff)
            self.w2 = torch.nn.Parameter(w2)

        if w3 is not None:
            assert w3.shape == (self.d_ff, self.d_model)
            self.w3 = torch.nn.Parameter(w3)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Inputs:
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # Generate frequency values for each dimension pair
        # For odd d_k dimensions, we have d_k/2 frequency values
        freqs = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        )

        # Generate position indices for all possible positions
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        # Compute the frequency matrix: outer product of positions and frequencies
        # Shape: (max_seq_len, d_k//2)
        # every line represents a R matrix
        freqs = torch.outer(t, freqs)

        # Precompute cos and sin values for efficiency
        # Shape: (max_seq_len, d_k//2)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.

        token positions is of shape (seq_len,) and contains the position indices for each token in the sequence. Because: given a token sequence, you have to also specify where each token is in the original text. Sometimes it is not from 0 to the end, maybe somewhere in the middle.

        cos_cached and sin_cached are of shape (max_seq_len, d_k//2), but we only need part of rows in the matrix, unless the seq_len=max_seq_len.
        """
        assert x.shape[-1] == self.d_k
        assert x.shape[-2] <= self.max_seq_len

        seq_len = x.shape[-2]
        assert token_positions.shape == (
            seq_len,
        ), f"token_positions must have shape ({seq_len},)"

        # Use token_positions to index into precomputed cos/sin tables
        cos = self.cos_cached[token_positions]  # Shape: (seq_len, d_k//2)
        print(cos.shape)
        assert cos.shape == (seq_len, self.d_k // 2)
        sin = self.sin_cached[token_positions]  # Shape: (seq_len, d_k//2)
        print(sin.shape)
        assert sin.shape == (seq_len, self.d_k // 2)

        # Reshape x to separate even and odd dimensions for rotation
        x_reshaped = x.view(*x.shape[:-1], -1, 2)  # (..., seq_len, d_k//2, 2)

        # Extract even (x1) and odd (x2) components
        x1 = x_reshaped[..., 0]  # (..., seq_len, d_k//2)
        x2 = x_reshaped[..., 1]  # (..., seq_len, d_k//2)

        # Apply rotation using the position-specific cos/sin values
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        # Combine and reshape back
        x_rot = torch.stack([x1_rot, x2_rot], dim=-1)
        return x_rot.view(*x.shape)


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.
    We use a trick for numerical stability: subtract the max value from the input tensor before exponentiating. This doesn't affect the result.

    Inputs:
        x: torch.Tensor Input tensor.
        dim: int Dimension along which to compute the softmax.

    Returns:
        torch.Tensor Softmax of the input tensor along the specified dimension.
    """
    # Subtract the max for numerical stability
    x_max = torch.max(x, dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    sum_e_x = torch.sum(e_x, dim=dim, keepdim=True)
    return e_x / sum_e_x


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the scaled dot-product attention.
    Each row of query and key represents the query vector and key vector for a token.

    Inputs:
        query: torch.Tensor of shape (batch_size, ..., seq_len_q, d_k)
        key: torch.Tensor of shape (batch_size, ..., seq_len_k, d_k)
        value: torch.Tensor of shape (batch_size, ..., seq_len_v, d_v)
        mask: torch.Tensor of shape (batch_size, 1, 1, seq_len_k) or None

    Returns:
        torch.Tensor of shape (seq_len, d_v)
    """
    d_k = query.size(-1)
    # query @ key.transpose is the dot product between each query and key
    # Each row of scores represents that vector's query with other vectors' keys
    scores = query @ key.transpose(-2, -1) / sqrt(d_k)  # Scaled dot-product

    if mask is not None:
        assert (
            mask.shape == scores.shape
        ), f"Mask shape {mask.shape} must match scores shape {scores.shape}"
        # fills 0 with -inf
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = softmax(scores, dim=-1)  # Softmax over the last dimension
    # every row of value matrix is the vector's row
    # every row of output is the weighted sum of all value vectors
    # seq_len rows corresponds to seq_len input vectors
    output = attn_weights @ value  # Weighted sum of values

    return output


class multihead_self_attention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding = None,
        device=None,
        dtype=None,
    ):
        """
        Inputs:
        d_model: int, Dimensionality of the Transformer block inputs.
        num_heads: int, Number of heads to use in multi-head self-attention.
        rope: RotaryPositionalEmbedding, Optional RoPE module for positional encoding.

        d_model is the length of each input vector resulted from embedding layer.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.rope = rope

        # Validate RoPE dimensions if provided
        if self.rope is not None:
            assert (
                self.rope.d_k == self.d_k
            ), f"RoPE d_k ({self.rope.d_k}) must match head dimension ({self.d_k})"

        # combined version
        self.w_qkv = nn.Parameter(
            torch.randn(3 * d_model, d_model, device=device, dtype=dtype)
        )
        self.w_o = nn.Parameter(
            torch.randn(d_model, d_model, device=device, dtype=dtype)
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inputs:
            x: torch.Tensor of shape (..., seq_len, d_model)
            token_positions: torch.Tensor of shape (seq_len,) containing position indices for RoPE
            mask: torch.Tensor of shape (..., seq_len, seq_len) or None

        Output:
            torch.Tensor of shape (..., seq_len, d_model)

        Each row of input x represents a token vector.
        """
        # Get all dimensions
        *leading_dims, seq_len, _ = x.shape
        assert (
            x.shape[-1] == self.d_model
        ), f"Input feature dimension {x.shape[-1]} must match d_model {self.d_model}"

        # If RoPE is used, token_positions must be provided
        if self.rope is not None:
            # if token_positions is not provided, generate default positions
            if token_positions is None:
                # Generate sequential positions from 0 to seq_len-1
                token_positions = torch.arange(
                    seq_len, device=x.device, dtype=torch.long
                )
            # Handle both single-dim and multi-dim token_positions
            if token_positions.dim() > 1:
                # Extract the last dimension: (..., seq_len) -> (seq_len,)
                token_positions = token_positions.view(-1)[-seq_len:]

            assert token_positions.shape == (
                seq_len,
            ), f"token_positions must have shape ({seq_len},), but now is {token_positions.shape}"

        # Single matrix multiplication for Q, K, V
        # Shape: (..., seq_len, 3 * d_model)
        qkv = x @ self.w_qkv.T
        # Split into Q, K, V
        # Each has shape: (..., seq_len, d_model)
        Q, K, V = qkv.chunk(3, dim=-1)

        # Reshape Q, K, V for multi-head attention
        # New shape: (..., num_heads, seq_len, d_k)
        Q = Q.view(*leading_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.view(*leading_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.view(*leading_dims, seq_len, self.num_heads, self.d_v).transpose(-3, -2)

        # Apply RoPE to Q and K if provided
        if self.rope is not None:
            # Apply RoPE to each head independently
            # Q and K shape: (..., num_heads, seq_len, d_k)
            Q_rotated = torch.zeros_like(Q)
            K_rotated = torch.zeros_like(K)

            for head_idx in range(self.num_heads):
                # Extract Q and K for this head: (..., seq_len, d_k)
                Q_head = Q[..., head_idx, :, :]
                K_head = K[..., head_idx, :, :]

                # Apply RoPE rotation
                Q_rotated[..., head_idx, :, :] = self.rope(Q_head, token_positions)
                K_rotated[..., head_idx, :, :] = self.rope(K_head, token_positions)

            Q = Q_rotated
            K = K_rotated

        # Create causal mask if none provided
        if mask is None:
            # Create lower triangular mask: 1s for positions that can attend, 0s for masked positions
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
            )
            # Expand to match the shape needed for attention scores
            # Shape: (..., num_heads, seq_len, seq_len)
            mask = causal_mask.expand(*leading_dims, self.num_heads, seq_len, seq_len)
        else:
            # If mask is provided, ensure it has the right shape and expand for heads
            expected_mask_shape = (*leading_dims, seq_len, seq_len)
            assert (
                mask.shape == expected_mask_shape
            ), f"Mask shape {mask.shape} must be {expected_mask_shape}"
            # Expand mask to match the number of heads
            mask = mask.unsqueeze(-3).expand(
                *leading_dims, self.num_heads, seq_len, seq_len
            )

        # Apply scaled dot-product attention
        attn_output = scaled_dot_product_attention(Q, K, V, mask=mask)
        # attn_output shape: (..., num_heads, seq_len, d_v)

        # Concatenate heads and reshape back to (..., seq_len, d_model)
        attn_output = (
            attn_output.transpose(-3, -2)
            .contiguous()
            .view(*leading_dims, seq_len, self.d_model)
        )

        # Final linear layer
        attn_output = attn_output @ self.w_o.T  # Shape: (..., seq_len, d_model)

        return attn_output

    def set_weights(
        self,
        q_proj_weight: torch.Tensor = None,
        k_proj_weight: torch.Tensor = None,
        v_proj_weight: torch.Tensor = None,
        o_proj_weight: torch.Tensor = None,
    ):
        """
        This function sets the weights of the multihead_self_attention layer.

        Args:
            q_proj_weight: torch.Tensor of shape (d_model, d_model) - Query projection weights
            k_proj_weight: torch.Tensor of shape (d_model, d_model) - Key projection weights
            v_proj_weight: torch.Tensor of shape (d_model, d_model) - Value projection weights
            o_proj_weight: torch.Tensor of shape (d_model, d_model) - Output projection weights
        """
        if (
            q_proj_weight is not None
            and k_proj_weight is not None
            and v_proj_weight is not None
        ):
            # Verify individual weight shapes
            assert q_proj_weight.shape == (self.d_model, self.d_model)
            assert k_proj_weight.shape == (self.d_model, self.d_model)
            assert v_proj_weight.shape == (self.d_model, self.d_model)

            # Concatenate Q, K, V weights into a single matrix
            # Shape: (3 * d_model, d_model)
            w_qkv_combined = torch.cat(
                [q_proj_weight, k_proj_weight, v_proj_weight], dim=0
            )
            self.w_qkv = torch.nn.Parameter(w_qkv_combined)

        if o_proj_weight is not None:
            assert o_proj_weight.shape == (self.d_model, self.d_model)
            self.w_o = torch.nn.Parameter(o_proj_weight)


class transformer_block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
        dtype=None,
    ):
        """
        Inputs:
            d_model: int Dimensionality of the Transformer block inputs.
            num_heads: int Number of heads to use in multi-head self-attention.
            d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        """
        super().__init__()
        self.RoPE = RotaryPositionalEmbedding(
            theta, d_k=d_k, max_seq_len=max_seq_len, device=device
        )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.attention = multihead_self_attention(
            d_model, num_heads, self.RoPE, device, dtype
        )
        self.ffn = SwiGLU(d_model, d_ff=d_ff)
        self.RMSNorm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.RMSNorm2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: torch.Tensor of shape (..., seq_len, d_model)

        Output:
            torch.Tensor of shape (..., seq_len, d_model)
        """
        x = self.attention(self.RMSNorm1(x)) + x
        x = self.ffn(self.RMSNorm2(x)) + x
        return x

    def set_weights(
        self,
        q_proj_weight: torch.Tensor,
        k_proj_weight: torch.Tensor,
        v_proj_weight: torch.Tensor,
        o_proj_weight: torch.Tensor,
        ln1_weight: torch.Tensor,
        w1_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w3_weight: torch.Tensor,
        ln2_weight: torch.Tensor,
    ):
        """Set all weights for this transformer block."""
        # Set attention weights
        self.attention.set_weights(
            q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight
        )

        # Set first RMSNorm weights
        self.RMSNorm1.set_weights(ln1_weight)

        # Set FFN weights
        self.ffn.set_weights(w1_weight, w2_weight, w3_weight)

        # Set second RMSNorm weights
        self.RMSNorm2.set_weights(ln2_weight)


class transformer_lm(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        d_k: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        device=None,
        dtype=None,
    ):
        """
        Additional inputs:
            vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
            context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
            num_layers: int The number of Transformer blocks to use.
        """
        super().__init__()
        # self.transformer_blocks includes num_layers transformer_block modules
        self.transformer_blocks = nn.ModuleList(
            [
                transformer_block(
                    d_model, num_heads, d_ff, theta, d_k, context_length, device, dtype
                )
                for _ in range(num_layers)
            ]
        )
        self.token_embedding = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.final_RMSNorm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: torch.Tensor of shape (batch_size, seq_len) where each value is an integer in [0, vocab_size)

        Output:
            torch.Tensor of shape (batch_size, seq_len, vocab_size)
        """
        x = self.token_embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_RMSNorm(x)
        x = self.lm_head(x)
        return x
