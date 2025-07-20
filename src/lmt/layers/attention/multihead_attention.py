# Copyright 2025 Michael Ellis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Multi-Head Attention Layer."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Layer."""

    # Correctly type self.mask as a Tensor
    mask: torch.Tensor

    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        num_heads: int,
        qkv_bias: bool = False,
    ):
        """Initialize the Multi-Head Attention layer.

        Instead of a single attention 'head' as in the SelfAttention or
        CausalAttention classes, MultiHeadAttention calculates attention
        weights using `num_heads` self-attention mechanisms each
        with causal masks in parallel. The output of the `num_heads`
        self-attention
        machanisms are concatenated together and applied to the queries
        to produce the context vectors.


        Args:
            embed_dim (int): Dimension of token embedding vectors.
            context_length (int): Max length of input sequence.
            num_heads (int): Number of attention heads to use.
            qkv_bias (bool): Whether to use bias in query, key, and value
                projections.
        """
        super().__init__()
        assert embed_dim % num_heads == 0, (
            'd_out must be divisible by num_heads'
        )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_query = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.W_value = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Multi-Head Attention layer.

        Args:
            x (torch.Tensor): Token embedding vectors with shape (batch_size,
                seq_length, embed_dim). Assumed to have position vectors
                added previously or no position vectors will be added.

        Returns:
            torch.Tensor: Context vector, a reweighting of the input token
                embedding vectors according to the multi-head scaled
                dot-product attention mechanism with a causal mask and the
                same dimension as the input (batch_size, seq_length,
                embed_dim).

        """
        b, seq_length, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, seq_length, self.num_heads, self.head_dim)
        queries = queries.view(b, seq_length, self.num_heads, self.head_dim)
        values = values.view(b, seq_length, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:seq_length, :seq_length]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] ** 0.5, dim=-1
        )

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, seq_length, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec
