"""Implementation of layer normalizer."""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization."""

    def __init__(self, embed_dim: int, eps: float = 1e-5):
        """Initialize LayerNorm with given embedding dimension and epsilon."""
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization to the input tensor."""
        mean_x = x.mean(dim=-1, keepdim=True)
        var_x = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean_x) / torch.sqrt(var_x + self.eps)
        return self.scale * norm_x + self.shift
