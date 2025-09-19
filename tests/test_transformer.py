"""Unit tests for the Transformer block."""

import torch
import torch.nn as nn

from lmt.layers.transformers import TransformerBlock
from lmt.models.config import ModelConfig, ModelConfigPresets


class TestTransformerBlock:
    """Test suite for the TransformerBlock."""

    def test_transformer_block_initialization(self):
        """Test TransformerBlock initializes correctly."""
        config = ModelConfigPresets.small_gpt()
        block = TransformerBlock(config)

        # Check if block has expected components
        assert hasattr(block, 'attn')
        assert hasattr(block, 'ff')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
        assert hasattr(block, 'dropout_shortcut')

        # Check layer norm dimensions
        assert block.norm1.normalized_shape == (config.embed_dim,)
        assert block.norm2.normalized_shape == (config.embed_dim,)

    def test_transformer_block_forward_shape(self):
        """Test that TransformerBlock forward pass preserves input shape."""
        config = ModelConfigPresets.small_gpt()
        block = TransformerBlock(config)

        batch_size = 2
        seq_length = 8
        input_tensor = torch.randn(batch_size, seq_length, config.embed_dim)

        output = block(input_tensor)

        assert output.shape == input_tensor.shape

    def test_transformer_block_residual_connections(self):
        """Test that residual connections work properly."""
        config = ModelConfig(
            context_length=16,
            vocab_size=1000,
            num_layers=1,
            num_heads=4,
            embed_dim=128,
            dropout=0.0,  # No dropout for cleaner residual testing
        )
        block = TransformerBlock(config)
        block.eval()  # Set to eval mode

        batch_size = 1
        seq_length = 4
        input_tensor = torch.randn(batch_size, seq_length, config.embed_dim)

        with torch.no_grad():
            # Store intermediate values to verify residual connections
            x1 = input_tensor.clone()

            # First residual connection (attention)
            shortcut1 = x1
            x1_norm = block.norm1(x1)
            x1_attn = block.attn(x1_norm)
            x1_dropout = block.dropout_shortcut(x1_attn)
            x1_res = x1_dropout + shortcut1

            # Second residual connection (feedforward)
            shortcut2 = x1_res
            x2_norm = block.norm2(x1_res)
            x2_ff = block.ff(x2_norm)
            x2_dropout = block.dropout_shortcut(x2_ff)
            expected_output = x2_dropout + shortcut2

            actual_output = block(input_tensor)

            torch.testing.assert_close(
                actual_output, expected_output, atol=1e-6, rtol=1e-5
            )

    def test_transformer_block_custom_ff_network(self):
        """Test TransformerBlock with custom feedforward network."""
        config = ModelConfigPresets.small_gpt()

        # Create custom feedforward network
        custom_ff = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 2),
            nn.ReLU(),
            nn.Linear(config.embed_dim * 2, config.embed_dim),
        )

        config.ff_network = custom_ff
        block = TransformerBlock(config)

        # Verify custom network is used
        assert block.ff is custom_ff

        # Test forward pass
        batch_size = 1
        seq_length = 4
        input_tensor = torch.randn(batch_size, seq_length, config.embed_dim)

        output = block(input_tensor)
        assert output.shape == input_tensor.shape

    def test_transformer_block_gradient_flow(self):
        """Test that gradients flow properly through TransformerBlock."""
        config = ModelConfigPresets.small_gpt()
        block = TransformerBlock(config)

        batch_size = 2
        seq_length = 4
        input_tensor = torch.randn(
            batch_size, seq_length, config.embed_dim, requires_grad=True
        )

        output = block(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed for all parameters
        for name, param in block.named_parameters():
            assert param.grad is not None, f'No gradient for {name}'
            assert not torch.isnan(param.grad).any(), (
                f'NaN gradient for {name}'
            )

        # Check input gradient
        assert input_tensor.grad is not None
        assert not torch.isnan(input_tensor.grad).any()

    def test_transformer_block_different_configs(self):
        """Test TransformerBlock with different configurations."""
        configs = [
            ModelConfig(
                embed_dim=64,
                num_heads=2,
                context_length=8,
                vocab_size=100,
                num_layers=1,
            ),
            ModelConfig(
                embed_dim=128,
                num_heads=4,
                context_length=16,
                vocab_size=1000,
                num_layers=1,
            ),
            ModelConfig(
                embed_dim=256,
                num_heads=8,
                context_length=32,
                vocab_size=5000,
                num_layers=1,
            ),
        ]

        for config in configs:
            block = TransformerBlock(config)
            batch_size = 1
            seq_length = min(4, config.context_length)
            input_tensor = torch.randn(
                batch_size, seq_length, config.embed_dim
            )

            output = block(input_tensor)
            assert output.shape == input_tensor.shape

    def test_transformer_block_deterministic_inference(self):
        """Test that TransformerBlock produces deterministic outputs in eval mode."""
        config = ModelConfigPresets.small_gpt()
        block = TransformerBlock(config)
        block.eval()

        batch_size = 1
        seq_length = 4
        input_tensor = torch.randn(batch_size, seq_length, config.embed_dim)

        with torch.no_grad():
            output1 = block(input_tensor)
            output2 = block(input_tensor)

        torch.testing.assert_close(output1, output2)

    def test_transformer_block_training_vs_eval_mode(self):
        """Test that TransformerBlock behaves differently in training vs eval mode."""
        config = ModelConfig(
            context_length=16,
            vocab_size=1000,
            num_layers=1,
            num_heads=4,
            embed_dim=128,
            dropout=0.5,  # High dropout to see difference
        )
        block = TransformerBlock(config)

        batch_size = 1
        seq_length = 4
        input_tensor = torch.randn(batch_size, seq_length, config.embed_dim)

        # Training mode
        block.train()
        with torch.no_grad():
            output_train1 = block(input_tensor)
            output_train2 = block(input_tensor)

        # Eval mode
        block.eval()
        with torch.no_grad():
            output_eval1 = block(input_tensor)
            output_eval2 = block(input_tensor)

        # In training mode with dropout, outputs should be different
        assert not torch.allclose(output_train1, output_train2, atol=1e-6)

        # In eval mode, outputs should be identical
        torch.testing.assert_close(output_eval1, output_eval2)
