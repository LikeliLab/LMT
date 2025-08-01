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

from lmt.layers.attention import MultiHeadAttention
from lmt.models.config import ModelConfig

from .ff import DefaultFeedForward


class TransformerBlock(nn.Module):
    """A single block of the Transformer architecture, inspired by GPT-2.

    This block encapsulates the core computations of a Transformer layer:
    Multi-Head Attention, followed by a FeedForward Network. Both sub-layers
    are equipped with residual connections and layer normalization, and dropout
    is applied to the outputs of each sub-layer before the residual addition.
    """

    def __init__(self, model_config: ModelConfig):
        """Initializes a Transformer block.

        Args:
            model_config (ModelConfig): Configuration object containing model
                parameters such as embedding dimension, number of heads,
                context length, and dropout rate.
        """
        super().__init__()
        self.attn = MultiHeadAttention(model_config=model_config)

        if model_config.ff_network:
            self.ff = model_config.ff_network
        else:
            self.ff = DefaultFeedForward(
                embed_dim=model_config.embed_dim,
                hidden_dim=4 * model_config.embed_dim,
            )

        self.norm1 = nn.LayerNorm(model_config.embed_dim)
        self.norm2 = nn.LayerNorm(model_config.embed_dim)
        self.dropout_shortcut = nn.Dropout(model_config.dropout)

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
