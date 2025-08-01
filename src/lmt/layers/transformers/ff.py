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

"""Implementations of feedforward network in Transformer block."""

import torch
import torch.nn as nn


class DefaultFeedForward(nn.Module):
    """Default feedforward network used in Transformer block.

    This is the default feedforward network that will be used in the
    Transformer block. Feel free to implement and experiment with your
    own.
    """

    def __init__(self, embed_dim: int, hidden_dim: int):
        """Initialize the DefaultFeedForward network.

        Args:
            embed_dim (int): The dimensionality of the input and output
                embedding vectors for each token position.
            hidden_dim (int): The dimensionality of the intermediate (hidden)
                layer within the feedforward network. This is commonly
                `4 * embed_dim` in many Transformer architectures.
        """
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the feedforward network to the input tensor.

        The input `x` is typically the output of the multi-head attention
        sub-layer, after layer normalization and residual connection.
        The FFN processes each token position independently.

        Args:
            x (torch.Tensor): The input tensor to the feedforward network.
                Expected shape is `(batch_size, sequence_length, embed_dim)`.

        Returns:
            torch.Tensor: The output tensor from the feedforward network,
                with the same shape as the input:
                `(batch_size, sequence_length, embed_dim)`.
        """
        return self.layers(x)
