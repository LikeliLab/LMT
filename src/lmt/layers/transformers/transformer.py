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
    """A single block of the Transformer architecture, inspired by GPT-2.

    This block encapsulates the core computations of a Transformer layer:
    Multi-Head Attention, followed by a FeedForward Network. Both sub-layers
    are equipped with residual connections and layer normalization, and dropout
    is applied to the outputs of each sub-layer before the residual addition.
    """

    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        num_heads: int,
        dropout: float,
        qkv_bias: bool = False,
        ff_network: nn.Module | None = None,
    ):
        """Initializes a Transformer block.

        Args:
            embed_dim (int): The dimensionality of the input and output token
                embedding vectors.
            context_length (int): The maximum sequence length the model is
                designed to handle. This determines the size of the causal
                attention mask.
            num_heads (int): The number of attention heads to use in the
                Multi-Head Attention mechanism. The `embed_dim` must be
                divisible by `num_heads`.
            dropout (float): The dropout rate to apply after the attention
                and feed-forward sub-layers, before the residual connection.
                A common value is 0.1.
            qkv_bias (bool, optional): If True, biases are added to the
                query, key, and value linear projections within the
                MultiHeadAttention module. Defaults to False.
            ff_network (nn.Module | None, optional): An instance of a custom
                feed-forward network (FFN) to be used within this Transformer
                block. This module should accept a tensor of shape
                `(batch_size, seq_length, embed_dim)` and return a tensor
                of the same shape. If `None`, a `DefaultFeedForward`
                instance will be constructed with `hidden_dim = 4 * embed_dim`
                and GELU activation function. Defaults to None.
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
        """Performs the forward pass of the Transformer Block.

        Args:
            x (torch.Tensor): The input tensor to the Transformer block.
                Expected shape is `(batch_size, seq_length, embed_dim)`.
                This tensor typically represents token embeddings, often
                combined with positional embeddings.

        Returns:
            torch.Tensor: The output tensor from the Transformer block,
                with the same shape as the input:
                `(batch_size, seq_length, embed_dim)`. This output
                can then be fed into the next Transformer block or a final
                prediction layer.
        """
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
