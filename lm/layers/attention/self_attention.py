"""Implementation of self attention mechanism."""

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Class to implement self attention mechanism."""

    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        """Initialize the self attention module.

        Args:
            d_in (int): Input dimension.
            d_out (int): Output dimension.
            qkv_bias (bool): Whether to use bias in query, key, and value
                projections.
        """
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        """Forward pass for self attention.

        Args:
            x (torch.Tensor): Input tensor of shape
                (batch_size, seq_len, d_in).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_out).
        """
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(-2, -1)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vec = attn_weights @ values
        return context_vec
