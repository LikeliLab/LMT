"""Implementation of GPT architecture with a few modifications.

This module provides a `GPT` class that constructs a decoder-only
transformer model, similar in architecture to the Generative Pre-trained
Transformer (GPT) models. It handles token and positional embeddings,
processes them through a series of transformer blocks, and projects the
final output to vocabulary-sized logits for language modeling tasks.
"""

import torch
import torch.nn as nn

from lmt.layers.transformers import TransformerBlock


class GPT(nn.Module):
    """A decoder-only Transformer model inspired by the GPT architecture.

    This class assembles a complete GPT-style model from its core components.
    It begins by creating a combined token and positional embedding for the
    input sequence. This embedding is then passed through a stack of
    `TransformerBlock` layers. Finally, a layer normalization is applied,
    followed by a linear layer to produce the final logits over the vocabulary.
    """

    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        num_heads: int,
        n_layers: int,
        dropout: float,
        qkv_bias: bool = False,
        ff_network: nn.Module | None = None,
    ):
        """Initializes the GPT model.

        Args:
            embed_dim (int): The dimensionality of the token and positional
                embeddings, and the hidden size of the transformer blocks.
            context_length (int): The maximum sequence length that the model
                can process. This defines the size of the positional embedding
                table.
            vocab_size (int): The total number of unique tokens in the
                vocabulary. This determines the size of the token embedding
                table and the output dimension of the final linear layer.
            num_heads (int): The number of attention heads within each
                transformer block. `embed_dim` must be divisible by this value.
            n_layers (int): The number of `TransformerBlock` layers to stack.
            dropout (float): The dropout rate applied to the embeddings.
            qkv_bias (bool, optional): If True, enables bias in the query, key,
                and value projections within the attention mechanism of each
                transformer block. Defaults to False.
            ff_network (nn.Module | None, optional): A custom feed-forward
                network to be used within each transformer block. If None, a
                default feed-forward network will be used. Defaults to None.
        """
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(context_length, embed_dim)
        self.dropout_embed = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    context_length=context_length,
                    num_heads=num_heads,
                    dropout=dropout,
                    qkv_bias=qkv_bias,
                    ff_network=ff_network,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)
        self.out_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the GPT model.

        This method processes an input tensor of token indices through the
        model to generate logits for the next token prediction.

        Args:
            in_idx (torch.Tensor): A tensor of token indices.
                Shape: `(batch_size, seq_length)`.

        Returns:
            torch.Tensor: The output logits from the model.
                Shape: `(batch_size, seq_length, vocab_size)`.
        """
        _, seq_length = in_idx.shape
        tok_embeds = self.tok_embed(in_idx)
        pos_embeds = self.pos_embed(
            torch.arange(seq_length, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.dropout_embed(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
