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
from lmt.models import ModelConfig


class GPT(nn.Module):
    """A decoder-only Transformer model inspired by the GPT architecture.

    This class assembles a complete GPT-style model from its core components.
    It begins by creating a combined token and positional embedding for the
    input sequence. This embedding is then passed through a stack of
    `TransformerBlock` layers. Finally, a layer normalization is applied,
    followed by a linear layer to produce the final logits over the vocabulary.
    """

    def __init__(self, model_config: ModelConfig):
        """Initializes the GPT model.

        Args:
            model_config (ModelConfig): Configuration object containing model
                hyperparameters such as vocabulary size, embedding dimension,
                context length, number of layers, and dropout rate.
        """
        super().__init__()
        self.tok_embed = nn.Embedding(
            model_config.vocab_size, model_config.embed_dim
        )
        self.pos_embed = nn.Embedding(
            model_config.context_length, model_config.embed_dim
        )
        self.dropout_embed = nn.Dropout(model_config.dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(model_config=model_config)
                for _ in range(model_config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(model_config.embed_dim)
        self.out_head = nn.Linear(
            model_config.embed_dim, model_config.vocab_size, bias=False
        )

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


class GPTClassifier(nn.Module):
    """A GPT model with a classification head.

    This class adapts the GPT architecture for classification tasks. It reuses
    the core transformer blocks from the base GPT model but replaces the
    language modeling head with a new linear layer for classification.
    """

    def __init__(self, model_config: ModelConfig, num_classes: int):
        """Initializes the GPTClassifier model.

        Args:
            model_config (ModelConfig): Configuration object containing model
                hyperparameters.
            num_classes (int): The number of classes for the classification
                task.
        """
        super().__init__()
        # Initialize a base GPT model to reuse its core components.
        # We will not use its final `out_head`.
        self.gpt = GPT(model_config)

        # The classification head is a simple linear layer that maps the
        # embedding dimension to the number of classification classes.
        self.classification_head = nn.Linear(
            model_config.embed_dim, num_classes
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass for classification.

        This method processes an input tensor, passes it through the GPT model,
        and uses the final token's embedding to produce classification logits.

        Args:
            in_idx (torch.Tensor): A tensor of token indices.
                Shape: `(batch_size, seq_length)`.

        Returns:
            torch.Tensor: The output logits from the classification head.
                Shape: `(batch_size, num_classes)`.
        """
        _, seq_length = in_idx.shape

        # Re-implement the forward pass up to the final hidden state to
        # bypass the language modeling head of the base GPT model.
        tok_embeds = self.gpt.tok_embed(in_idx)
        pos_embeds = self.gpt.pos_embed(
            torch.arange(seq_length, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.gpt.dropout_embed(x)
        x = self.gpt.transformer_blocks(x)
        x = self.gpt.final_norm(x)

        # For classification, we extract the hidden state of a single token
        # to represent the entire sequence. The hidden state of the last token
        # is a common and effective choice for this purpose.
        # `x` has shape `(batch_size, seq_length, embed_dim)`.
        last_token_hidden_state: torch.Tensor = x[:, -1, :]

        # Pass the pooled hidden state through the classification head
        # to get the final classification logits.
        classification_logits: torch.Tensor = self.classification_head(
            last_token_hidden_state
        )

        return classification_logits
