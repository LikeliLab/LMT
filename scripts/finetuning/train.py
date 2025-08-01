"""Script for finetuning LLM for classification."""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import tensorflow as tf
import torch

from lmt.datasets import ClassificationDataset
from lmt.models.config import ModelConfig
from lmt.models.gpt import GPT
from lmt.tokenizer.bpe import BPETokenizer

# Import your custom modules
from scripts.finetuning.utils import (
    create_balanced_dataset,
    download_and_unzip_spam_data,
    download_model,
    load_gpt2_params_from_tf_ckpt,
    load_weights_into_gpt,
    random_split,
)


def main(args):
    """Main function  for finetuning LLM for classification."""
    # --- 1. Download Model ---
    print('--- Step 1: Download Model ---')
    if args.download_model:
        download_model(model_size='124M')

    print('--- Step 2: Set Model Config ---')
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

    print('--- Step 3: Load Model Pretrained Weights ---')
    model_dir = 'scripts/finetuning/models/124M/'
    # load tf model
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)

    with open(os.path.join(model_dir, 'hparams.json'), encoding='utf-8') as f:
        settings = json.load(f)

    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # type: ignore
    load_weights_into_gpt(model, params)
    print('Model weights loaded')

    # --- 2. Calculate and Display Total Parameters ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {total_params:,}')

    # --- 3. Setup Paths and Directories from Arguments ---
    print('--- Step 3: Setting up paths ---')
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
    print('\n--- Step 4: Downloading and preparing raw data ---')
    download_and_unzip_spam_data(
        zip_path=zip_path,
        extracted_path=extracted_path,
        data_file_path=data_file_path,
    )

    # --- 5. Load, Preprocess, and Split Data ---
    print('\n--- Step 5: Loading, preprocessing, and splitting data ---')
    df = pd.read_csv(
        data_file_path, sep='\t', header=None, names=['Label', 'Text']
    )

    # Create a balanced dataset
    balanced_df = create_balanced_dataset(df)

    # Use a lambda function to be more explicit for the type checker:
    balanced_df['Label'] = balanced_df['Label'].replace({'ham': 0, 'spam': 1})

    # Split into train, validation, and test sets based on fractions from args
    print(
        f'Splitting data: {args.train_ratio * 100}% train, '
        f'{args.val_ratio * 100}% validation'
    )
    train_df, validation_df, test_df = random_split(
        balanced_df,
        train_frac=args.train_ratio,
        validation_frac=args.val_ratio,
    )

    # Save the splits to CSV files
    train_df.to_csv(train_csv_path, index=False)
    validation_df.to_csv(validation_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print('Data split and saved:')

    # --- 4. Create Datasets and DataLoaders ---
    print('\n--- Step 6: Creating PyTorch Datasets and DataLoaders ---')

    # Initialize the tokenizer based on the argument
    tokenizer = BPETokenizer('gpt2', allowed_special={'<|endoftext|>'})

    train_dataset = ClassificationDataset(
        csv_file=args.data_dir + '/train.csv',
        max_length=None,
        tokenizer=tokenizer,
    )
    print(f'Max Sequence Length = {train_dataset.max_length}')

    # val_dataset = ClassificationDataset(
    #     csv_file=args.data_dir + '/validation.csv',
    #     max_length=train_dataset.max_length,
    #     tokenizer=tokenizer,
    # )
    # test_dataset = ClassificationDataset(
    #     csv_file=args.data_dir + '/test.csv',
    #     max_length=train_dataset.max_length,
    #     tokenizer=tokenizer,
    # )

    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    # )

    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=0,
    #     drop_last=False,
    # )

    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=0,
    #     drop_last=False,
    # )


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
