from math import sqrt
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
