"""Implementation of transformer block for GPT2 model."""

import torch
import torch.nn as nn
from attention import MultiHeadAttention
from normalization import LayerNorm


class TransformerBlock(nn.Module):
    """Implementation of transformer block for GPT2 model."""

    def __init__(self, cfg: dict):
        """Initialize the Transformer Block."""
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg['embed_dim'],
            d_out=cfg['embed_dim'],
            context_length=cfg['context_length'],
            dropout=cfg['dropout'],
            num_heads=cfg['num_heads'],
            qkv_bias=cfg.get('qkv_bias', False),
        )

        self.ff = nn.Sequential(
            nn.Linear(cfg['embed_dim'], 4 * cfg['embed_dim']),
            nn.GELU(),
            nn.Linear(4 * cfg['embed_dim'], cfg['embed_dim']),
        )

        self.norm1 = LayerNorm(cfg['embed_dim'])
        self.norm2 = LayerNorm(cfg['embed_dim'])
        self.dropout_shortcut = nn.Dropout(cfg['dropout'])

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
