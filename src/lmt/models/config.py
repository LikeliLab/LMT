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
"""This module defines the configuration for the model architecture."""

from dataclasses import dataclass

from torch.nn import Module


@dataclass
class ModelConfig:
    """Configuration for the model architecture.

    Attributes:
        context_length (int): Max sequence length (context window).
        vocab_size (int): Vocabulary size, e.g., GPT-2 vocab size.
        num_layers (int): Number of transformer blocks.
        num_heads (int): Number of attention heads.
        embed_dim (int): Embedding dimension.
        dropout (float): Probability of a weight to be zeroed. Regularization
            technique to prevent overfitting. Must be a float between 0.0
            and 1.0.
        qkv_bias (bool, optional): If True, enables bias in the query, key,
                and value projections within the attention mechanism of each
                transformer block. Defaults to False.
        ff_network (nn.Module | None, optional): A custom feed-forward
            network to be used within each transformer block. If None, a
            default feed-forward network will be used. Defaults to None.
    """

    context_length: int = 4
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    dropout: float = 0.1
    qkv_bias: bool = False
    ff_network: Module | None = None
