"""Script for finetuning LLM for classification."""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lmt.models.config import ModelConfig
from lmt.models.gpt import GPT
from lmt.tokenizer.bpe import BPETokenizer
from lmt.training.datasets import ClassificationDataset

# Import your custom modules
from scripts.finetuning.utils import (
    create_balanced_dataset,
    download_and_unzip_spam_data,
    download_model,
    load_gpt2_params_from_tf_ckpt,
    load_weights_into_gpt,
    plot_metrics,
    random_split,
    train,
)


def main(args):
    """Main function  for finetuning LLM for classification."""
    # Create the save directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Download Model ---
    print('\n--- Step 1: Download Model ---')
    if args.download_model:
        download_model(model_size='124M')

    print('\n--- Step 2: Set Model Config ---')
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
        qkv_bias=args.qkv_bias,
        ff_network=None,
    )

    model = GPT(model_config).to(device)

    print('\n--- Step 3: Load Model Pretrained Weights ---')
    model_dir = 'scripts/finetuning/models/124M/'
    # load tf model
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    with open(os.path.join(model_dir, 'hparams.json'), encoding='utf-8') as f:
        settings = json.load(f)

    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # type: ignore
    load_weights_into_gpt(model, params)
    print('Model weights loaded')

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params:,}')

    print('Freeze parameter weights')
    for param in model.parameters():
        param.requires_grad = False

    print('Replace output layer with a classification head')
    model.out_head = nn.Linear(model_config.embed_dim, 2, bias=True)

    print('Set the final LayerNorm and Last Transformer block to traninable')
    for param in model.transformer_blocks[-1].parameters():
        param.requires_grad = True

    model.to(device)

    # --- 3. Setup Paths and Directories from Arguments ---
    print('\n--- Step 4: Setting up paths ---')
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    # Define paths for raw data and processed splits
    zip_path = data_dir / 'sms_spam_collection.zip'
    extracted_path = Path('sms_spam_collection')
    data_file_path = data_dir / 'SMSSpamCollection.tsv'

    train_csv_path = data_dir / 'train.csv'
    validation_csv_path = data_dir / 'validation.csv'
    test_csv_path = data_dir / 'test.csv'

    print(f'Data directory set to: {data_dir.resolve()}')

    # --- 4. Download and Prepare Raw Data ---
    print('\n--- Step 5: Downloading and preparing raw data ---')
    download_and_unzip_spam_data(
        zip_path=zip_path,
        extracted_path=extracted_path,
        data_file_path=data_file_path,
    )

    # --- 5. Load, Preprocess, and Split Data ---
    print('\n--- Step 6: Loading, preprocessing, and splitting data ---')
    df = pd.read_csv(
        data_file_path, sep='\t', header=None, names=['Label', 'Text']
    )

    # Create a balanced dataset
    balanced_df = create_balanced_dataset(df)

    # Use a lambda function to be more explicit for the type checker:
    balanced_df['Label'] = balanced_df['Label'].map({'ham': 0, 'spam': 1})  # type: ignore

    # Split into train, validation, and test sets based on fractions from args
    print(
        f'Splitting data: {args.train_ratio * 100}% train, '
        f'{args.val_ratio * 100}% validation, '
        f'{(1 - args.train_ratio - args.val_ratio) * 100:.1f}% test'
    )
    train_df, validation_df, test_df = random_split(
        balanced_df,
        train_frac=args.train_ratio,
        validation_frac=args.val_ratio,
    )

    print(f'Train shape: {train_df.shape}')
    print(f'Val shape: {validation_df.shape}')
    print(f'Test shape: {test_df.shape}')

    # Save the splits to CSV files
    train_df.to_csv(train_csv_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print('Data split and saved:')

    # --- 4. Create Datasets and DataLoaders ---
    print('\n--- Step 7: Creating PyTorch Datasets and DataLoaders ---')

    # Initialize the tokenizer based on the argument
    tokenizer = BPETokenizer('gpt2', allowed_special={'<|endoftext|>'})

    train_dataset = ClassificationDataset(
        csv_file=args.data_dir + '/train.csv',
        max_length=None,
        tokenizer=tokenizer,
    )
    print(f'Max Sequence Length = {train_dataset.max_length}')

    val_dataset = ClassificationDataset(
        csv_file=args.data_dir + '/validation.csv',
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )
    # test_dataset = ClassificationDataset(
    #     csv_file=args.data_dir + '/test.csv',
    #     max_length=train_dataset.max_length,
    #     tokenizer=tokenizer,
    # )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        drop_last=False,
    )

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=0,
    #     drop_last=False,
    # )

    print('\n--- Step 8: Train ---')
    start_time = time.time()
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, weight_decay=0.1
    )

    num_epochs = 5
    eval_freq = 50
    train_losses, val_losses, train_accs, val_accs, examples_seen = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=eval_freq,
        eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f'Training completed in {execution_time_minutes:.2f} minutes.')

    # --- 5. Plotting Losses ---
    plot_metrics(
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        eval_freq=eval_freq,
        save_dir=args.save_dir,
    )

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
        description=(
            'Download, process, and prepare the SMS Spam Collection '
            'dataset for training.'
        )
    )

    # --- Argument Groups ---
    data_group = parser.add_argument_group('Data and Save Paths')
    model_group = parser.add_argument_group('Model Hyperparameters')
    training_group = parser.add_argument_group('Training Parameters')

    # --- Data and Path Arguments ---
    data_group.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help=(
            'Directory to save the dataset and processed files. '
            '(Default: data)'
        ),
    )
    data_group.add_argument(
        '--save_dir',
        type=str,
        default='scripts/finetuning/runs',
        help='Directory to save model checkpoints.',
    )

    # --- Model Group ---
    model_group.add_argument(
        '--download_model',
        type=bool,
        default=False,
        help='Download pretrained GPT model.',
    )
    model_group.add_argument(
        '--context_length',
        type=int,
        default=1024,
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
        '--dropout', type=float, default=0.0, help='Dropout rate.'
    )
    model_group.add_argument(
        '--qkv_bias', type=bool, default=True, help='Add bias for qkv matrices'
    )

    # --- Training Parameter Arguments ---
    training_group.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Proportion of data to use for training.',
    )
    training_group.add_argument(
        '--val_ratio',
        type=float,
        default=0.2,
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
