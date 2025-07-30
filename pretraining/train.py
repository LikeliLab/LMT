"""Script for training LLM."""

import argparse
from pathlib import Path

import torch

from lmt.models.config import ModelConfig
from lmt.models.gpt import GPT
from lmt.tokenizer.bpe import BPETokenizer
from pretraining.utils import (
    create_dataloader,
    plot_losses,
    read_text_file,
    train,
)


def main(args):
    """Main function to orchestrate the model training pipeline."""
    # --- 1. Setup and Configuration ---

    # Create the save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set the device for training (e.g., 'cuda', 'mps', 'cpu')
    device = torch.device(args.device)

    # Instantiate configuration objects from command-line arguments
    model_config = ModelConfig(
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        qkv_bias=False,
        ff_network=None,
    )

    # --- 2. Data Preparation ---

    # Read and split the dataset
    text_data = read_text_file(Path(args.data_file))
    split_idx = int(args.train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # Initialize tokenizer
    tokenizer = BPETokenizer('gpt2', allowed_special={'<|endoftext|>'})

    # Create DataLoaders for training and validation sets
    train_dataloader = create_dataloader(
        train_data,
        batch_size=args.batch_size,
        max_length=model_config.context_length,
        stride=model_config.context_length,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        tokenizer=tokenizer,
    )

    val_dataloader = create_dataloader(
        val_data,
        batch_size=args.batch_size,
        max_length=model_config.context_length,
        stride=model_config.context_length,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        tokenizer=tokenizer,
    )

    # --- 3. Model and Optimizer Initialization ---

    # Initialize the GPT model and move it to the specified device
    model = GPT(model_config).to(device)

    # Initialize the AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # --- 4. Training ---

    print('Starting model training...')
    train_losses, val_losses, track_tokens_seen = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        eval_iter=args.eval_iter,
        start_context=args.start_context,
        tokenizer=tokenizer,
    )
    print('Training finished.')

    # --- 5. Plotting Losses ---
    plot_losses(save_dir, train_losses, val_losses, track_tokens_seen)

    # --- 6. Save the Final Model ---

    model_save_path = save_dir / 'model_and_optimizer.pth'
    print(f'Saving model and optimizer to {model_save_path}')
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        model_save_path,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Robust script for training a GPT model.'
    )

    # --- Argument Groups ---
    data_group = parser.add_argument_group('Data and Save Paths')
    model_group = parser.add_argument_group('Model Hyperparameters')
    training_group = parser.add_argument_group('Training Parameters')

    # --- Data and Save Path Arguments ---
    data_group.add_argument(
        '--data_file',
        type=str,
        default='pretraining/data/the-verdict.txt',
        help='Path to the input text file.',
    )
    data_group.add_argument(
        '--save_dir',
        type=str,
        default='pretraining/runs',
        help='Directory to save model checkpoints.',
    )

    # --- Model Hyperparameter Arguments ---
    model_group.add_argument(
        '--context_length',
        type=int,
        default=256,
        help='Maximum context length for the model.',
    )
    model_group.add_argument(
        '--vocab_size', type=int, default=50257, help='Vocabulary size.'
    )
    model_group.add_argument(
        '--num_layers',
        type=int,
        default=12,
        help='Number of transformer layers.',
    )
    model_group.add_argument(
        '--num_heads', type=int, default=12, help='Number of attention heads.'
    )
    model_group.add_argument(
        '--embed_dim', type=int, default=768, help='Embedding dimension.'
    )
    model_group.add_argument(
        '--dropout', type=float, default=0.1, help='Dropout rate.'
    )

    # --- Training Parameter Arguments ---
    training_group.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='Proportion of data to use for training.',
    )
    training_group.add_argument(
        '--batch_size', type=int, default=2, help='Batch size for training.'
    )
    training_group.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4,
        help='Learning rate for the optimizer.',
    )
    training_group.add_argument(
        '--weight_decay',
        type=float,
        default=0.1,
        help='weight decay for the optimizer.',
    )
    training_group.add_argument(
        '--num_epochs',
        type=int,
        default=10,
        help='Total number of epochs to train for.',
    )
    training_group.add_argument(
        '--eval_freq',
        type=int,
        default=5,
        help='Frequency (in steps) to perform evaluation.',
    )
    training_group.add_argument(
        '--eval_iter',
        type=int,
        default=5,
        help='Number of batches to use for evaluation.',
    )
    training_group.add_argument(
        '--device',
        type=str,
        default='mps',
        help="Device to use for training (e.g., 'cuda', 'mps', 'cpu').",
    )
    training_group.add_argument(
        '--start_context',
        type=str,
        default='Every effort moves you',
        help='Starting context for text generation samples.',
    )

    # Parse the arguments and run the main function
    args = parser.parse_args()
    main(args)
