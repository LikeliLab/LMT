"""Implementation of GPT2 architecture with a few modifications."""

import torch
import torch.nn as nn

from lm.layers.normalization import LayerNorm
from lm.layers.transformers import TransformerBlock


class GPT2(nn.Module):
    """Implementation of GPT2 architecture with a few modifications."""

    def __init__(self, cfg):
        """Initialize the GPT2 model."""
        super().__init__()
        self.tok_embed = nn.Embedding(cfg['vocab_size'], cfg['embed_dim'])
        self.pos_embed = nn.Embedding(cfg['context_length'], cfg['embed_dim'])
        self.dropout_embed = nn.Dropout(cfg['dropout'])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['embed_dim'])
        self.out_head = nn.Linear(
            cfg['embed_dim'], cfg['vocab_size'], bias=False
        )

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GPT2 model."""
        batch_size, seq_length = in_idx.shape
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
