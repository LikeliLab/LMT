"""Implementation of GELU activation function."""

import torch
import torch.nn as nn


class GELU(nn.Module):
    """Implementation of GELU activation function."""

    def __init__(self):
        """Initialize the GELU activation function."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for GELU activation."""
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
