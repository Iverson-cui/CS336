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
