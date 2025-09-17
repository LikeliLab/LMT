"""Unit tests for the model utils module."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from lmt.models.utils import save_model_checkpoint


def test_save_model_checkpoint():
    """Test saving a model checkpoint creates the expected file."""
    # Create a simple model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create a simple config object
    class Config:
        learning_rate = 0.01
        epochs = 100

    config = Config()

    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / 'test_checkpoint.pth'

        # Save the checkpoint
        save_model_checkpoint(model, optimizer, config, save_path, epoch=5)

        # Verify the file was created
        assert save_path.exists()

        # Verify the checkpoint contains expected keys
        checkpoint = torch.load(save_path)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'config' in checkpoint
        assert 'epoch' in checkpoint
        assert checkpoint['epoch'] == 5
