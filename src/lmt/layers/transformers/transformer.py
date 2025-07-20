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

"""Implementation of transformer block for GPT2 model."""

import torch
import torch.nn as nn
from attention import MultiHeadAttention

from .ff import DefaultFeedForward


class TransformerBlock(nn.Module):
    """Implementation of transformer block for GPT2 model."""

    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        num_heads: int,
        dropout: float,
        qkv_bias: bool = False,
        ff_network: nn.Module | None = None,
    ):
        """Initialize the Transformer Block.

        Args:
            embed_dim (int): Dimension of token embedding vectors.
            context_length (int): Max length of input sequence.
            num_heads (int): Number of attention heads to use.
            dropout (float): Dropout rate for embedding features.
            qkv_bias (bool): Whether to use bias in query, key, and value
                projections.
            ff_network (nn.Module): Feed forward network to use within the
                tranformer block.

        """
        super().__init__()
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            context_length=context_length,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        if ff_network:
            self.ff = ff_network
        else:
            self.ff = DefaultFeedForward(
                embed_dim=embed_dim, hidden_dim=4 * embed_dim
            )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_shortcut = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Transformer Block."""
        # Multi-Head Attention
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        # Feed Forward Network
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout_shortcut(x)
        x = x + shortcut

        return x
