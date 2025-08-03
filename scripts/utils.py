# Copyright 2025 Michael Ellis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Supporting functions for model training.

This module provides comprehensive utilities for training language models,
specifically GPT-style transformers. It includes functionality for data
preparation, model weight loading, and dataset processing for both pretraining
and classification tasks.

The module supports:
    - Pretraining data preparation from text files
    - Classification data preparation from labeled datasets
    - GPT-2 model weight loading from TensorFlow checkpoints
    - SMS spam detection dataset processing
    - Model configuration for fine-tuning tasks

Example:
    Basic usage for preparing pretraining data:

    ```python
    args = TrainingArgs(
        task='pretraining', data_dir='data/', batch_size=32, context_length=512
    )
    dataloaders = prepare_data(args)
    ```

Dependencies:
    - PyTorch for model operations
    - TensorFlow for checkpoint loading
    - pandas for data manipulation
    - numpy for numerical operations
    - BPE tokenizer for text processing
"""

import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lmt.models.gpt import GPT
from lmt.training.dataloaders import (
    BPETokenizer,
    create_classification_dataloader,
    create_pretraining_dataloader,
)


def read_file(file_path: Path) -> str:
    """Reads the content from a .txt file.

    Args:
        file_path: A Path object pointing to the .txt file to be read.

    Returns:
        The complete content of the file as a string.

    Raises:
        ValueError: If the file is not a .txt file (checked by extension).
        FileNotFoundError: If the file does not exist at the specified path.

    Example:
        ```python
        content = read_file(Path('data/sample.txt'))
        print(f'File contains {len(content)} characters')
        ```
    """
    if file_path.suffix.lower() != '.txt':
        raise ValueError(
            f"Error: The file '{file_path.name}' is not a .txt file."
        )

    try:
        with file_path.open('r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Error: Could not find the file at path: '{file_path}'"
        ) from err
    except Exception as e:
        print(f'An unexpected error occurred: {e}')
        raise


def prepare_data(args) -> dict[DataLoader, DataLoader]:
    """Prepares data loaders based on the specified task type.

    This function serves as a dispatcher that routes to the appropriate data
    preparation function based on the task specified in args.

    Args:
        args: Training arguments object containing task configuration.
            Must have a 'task' attribute that is either 'pretraining' or
            'classification'.

    Returns:
        A dictionary containing DataLoader objects with keys:
            - 'train_dataloader': Training data loader
            - 'val_dataloader': Validation data loader

    Raises:
        ValueError: If the task specified in args is not supported.

    Example:
        ```python
        args = TrainingArgs(task='pretraining', data_dir='data/')
        dataloaders = prepare_data(args)
        train_loader = dataloaders['train_dataloader']
        val_loader = dataloaders['val_dataloader']
        ```
    """
    task = args.task
    match task:
        case 'pretraining':
            dataloaders = prepare_data_pretraining(args)
        case 'classification':
            dataloaders = prepare_data_classification(args)
        case _:  # default case
            raise ValueError(f'Invalid task: {task}')

    return dataloaders


def prepare_data_pretraining(args) -> dict[DataLoader, DataLoader]:
    """Prepares data loaders for language model pretraining tasks.

    This function loads a text file, splits it into training and validation
    sets, and creates PyTorch DataLoader objects for both sets using BPE
    tokenization.

    Args:
        args: Training arguments object containing:
            - data_dir: Directory containing the text data
            - train_ratio: Fraction of data to use for training (e.g., 0.8)
            - batch_size: Number of samples per batch
            - context_length: Maximum sequence length for tokenization
            - num_workers: Number of worker processes for data loading

    Returns:
        A dictionary containing DataLoader objects:
            - 'train_dataloader': Training data loader with shuffling enabled
            - 'val_dataloader': Validation data loader without shuffling

    Note:
        The function expects a file named 'the-verdict.txt' in the specified
        data directory. The data is split sequentially based on train_ratio.

    Example:
        ```python
        args = PretrainingArgs(
            data_dir='data/',
            train_ratio=0.8,
            batch_size=32,
            context_length=512,
            num_workers=4,
        )
        dataloaders = prepare_data_pretraining(args)
        ```
    """
    dataloaders = {}
    file_path = 'scripts' / Path(args.data_dir) / 'the-verdict.txt'
    print(f'file_path = {file_path}')
    text_data = read_file(file_path)
    split_idx = int(args.train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    tokenizer = BPETokenizer('gpt2', allowed_special={'<|endoftext|>'})

    dataloaders['train_dataloader'] = create_pretraining_dataloader(
        train_data,
        batch_size=args.batch_size,
        max_length=args.context_length,
        stride=args.context_length,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
    )

    dataloaders['val_dataloader'] = create_pretraining_dataloader(
        val_data,
        batch_size=args.batch_size,
        max_length=args.context_length,
        stride=args.context_length,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
    )

    return dataloaders


def prepare_data_classification(args) -> dict[DataLoader, DataLoader]:
    """Prepares data loaders for text classification tasks.

    This function downloads and processes the SMS Spam Collection dataset,
    creates balanced train/validation/test splits, and returns PyTorch
    DataLoader objects for training and validation.

    Args:
        args: Training arguments object containing:
            - data_dir: Directory to store downloaded and processed data
            - train_ratio: Fraction of data for training
            - val_ratio: Fraction of data for validation
            - batch_size: Number of samples per batch
            - num_workers: Number of worker processes for data loading

    Returns:
        A dictionary containing DataLoader objects:
            - 'train_dataloader': Training data loader with shuffling
            - 'val_dataloader': Validation data loader without shuffling

    Note:
        The function automatically downloads the SMS Spam Collection dataset
        if not already present. It creates a balanced dataset by downsampling
        the majority class ('ham') to match the minority class ('spam').

    Example:
        ```python
        args = ClassificationArgs(
            data_dir='data/',
            train_ratio=0.7,
            val_ratio=0.15,
            batch_size=32,
            num_workers=4,
        )
        dataloaders = prepare_data_classification(args)
        ```
    """
    dataloaders = {}
    data_dir = Path(args.data_dir)
    data_dir.mkdir(exist_ok=True)

    # Define paths for raw data and processed splits
    zip_path = data_dir / 'sms_spam_collection.zip'
    extracted_path = Path('sms_spam_collection')
    data_file_path = data_dir / 'SMSSpamCollection.tsv'

    train_csv_path = data_dir / 'train.csv'
    validation_csv_path = data_dir / 'validation.csv'
    test_csv_path = data_dir / 'test.csv'

    # --- Download and Prepare Raw Data ---
    print('\n --- Downloading and preparing raw data ---')
    download_and_unzip_spam_data(
        zip_path=zip_path,
        extracted_path=extracted_path,
        data_file_path=data_file_path,
    )

    print('\n --- Balance and split data ---')
    balance_and_split_data(
        data_file_path,
        train_csv_path,
        validation_csv_path,
        test_csv_path,
        args,
    )

    tokenizer = BPETokenizer('gpt2', allowed_special={'<|endoftext|>'})

    dataloaders['train_dataloader'] = create_classification_dataloader(
        csv_file=Path(train_csv_path),
        batch_size=args.batch_size,
        max_length=None,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
    )

    dataloaders['val_dataloader'] = create_classification_dataloader(
        csv_file=Path(validation_csv_path),
        batch_size=args.batch_size,
        max_length=None,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
    )

    return dataloaders


# =============================================================================
# ======== Classification utils ===============================================
# =============================================================================


def balance_and_split_data(
    data_file_path, train_csv_path, validation_csv_path, test_csv_path, args
):
    """Balances class distribution and splits data into train/validation/test.

    This function reads the SMS spam dataset, creates a balanced dataset by
    downsampling the majority class, converts string labels to integers,
    and splits the data into three sets based on the provided ratios.

    Args:
        data_file_path: Path to the input TSV file containing the raw dataset.
        train_csv_path: Path where the training CSV will be saved.
        validation_csv_path: Path where the validation CSV will be saved.
        test_csv_path: Path where the test CSV will be saved.
        args: Training arguments object containing:
            - train_ratio: Fraction of data for training
            - val_ratio: Fraction of data for validation

    Note:
        The function converts string labels ('ham', 'spam') to integers (0, 1)
        and prints the shape of each split for verification.

    Example:
        ```python
        balance_and_split_data(
            'data/SMSSpamCollection.tsv',
            'data/train.csv',
            'data/validation.csv',
            'data/test.csv',
            args,
        )
        ```
    """
    df = pd.read_csv(
        data_file_path, sep='\t', header=None, names=['Label', 'Text']
    )

    # Create a balanced dataset
    balanced_df = create_balanced_dataset(df)

    # Use a lambda function to be more explicit for the type checker:
    balanced_df['Label'] = balanced_df['Label'].map({'ham': 0, 'spam': 1})  # type: ignore

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


def download_and_unzip_spam_data(
    url: str = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip',
    zip_path: Path = Path('data/sms_spam_collection.zip'),
    extracted_path: Path = Path('sms_spam_collection'),
    data_file_path: Path = Path('data/SMSSpamCollection.tsv'),
) -> None:
    """Downloads, extracts, and renames the SMS Spam Collection dataset.

    This function handles the complete pipeline of acquiring the SMS Spam
    Collection dataset from the UCI Machine Learning Repository. It checks if
    the final data file already exists to avoid redundant downloads.

    Args:
        url: The URL to download the zip file from. Defaults to the official
            UCI repository URL for the SMS Spam Collection dataset.
        zip_path: The local path where the downloaded zip file will be saved.
            Defaults to 'data/sms_spam_collection.zip'.
        extracted_path: The directory where the zip file contents will be
            extracted. Defaults to 'sms_spam_collection'.
        data_file_path: The final path for the dataset file with .tsv
            extension. Defaults to 'data/SMSSpamCollection.tsv'.

    Returns:
        None. Files are downloaded and saved to the specified paths.

    Note:
        This function is adapted from the LLMs-from-scratch repository by
        Sebastian Raschka. It includes progress indication during download
        and handles file renaming to add the .tsv extension.

    Example:
        ```python
        # Use default parameters
        download_and_unzip_spam_data()

        # Use custom paths
        download_and_unzip_spam_data(
            zip_path=Path('custom/spam_data.zip'),
            data_file_path=Path('custom/spam_data.tsv'),
        )
        ```
    """
    if data_file_path.exists():
        print(
            f'{data_file_path} already exists.'
            f'Skipping download and extraction.'
        )
        return

    # Downloading the file
    print(f'Downloading data from {url}...')
    with (
        urllib.request.urlopen(url) as response,
        open(zip_path, 'wb') as out_file,
    ):
        out_file.write(response.read())
    print(f'Downloaded and saved to {zip_path}')

    # Unzipping the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)

    # Add .tsv file extension
    original_file_path = Path(extracted_path) / 'SMSSpamCollection'
    os.rename(original_file_path, data_file_path)
    print(f'File downloaded and saved as {data_file_path}')


def create_balanced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Creates a balanced dataset by downsampling the majority class.

    This function addresses class imbalance in the SMS Spam Collection dataset
    by randomly sampling from the majority class ('ham') to match the count of
    the minority class ('spam'). This ensures equal representation of both
    classes.

    Args:
        df: The input DataFrame containing the SMS dataset. Must have a 'Label'
            column with 'spam' and 'ham' values.

    Returns:
        A new balanced DataFrame with equal numbers of 'spam' and 'ham'
        samples. The returned DataFrame maintains the same column structure as
        the input.

    Note:
        This function uses a fixed random state (123) for reproducibility.
        The function is adapted from the LLMs-from-scratch repository.

    Example:
        ```python
        # Original dataset might have 4825 ham and 747 spam messages
        original_df = pd.read_csv('spam_data.csv')
        balanced_df = create_balanced_dataset(original_df)
        # Result: 747 ham and 747 spam messages
        ```
    """
    # Count the instances of "spam"
    num_spam = df[df['Label'] == 'spam'].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df['Label'] == 'ham'].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df['Label'] == 'spam']])

    return pd.DataFrame(balanced_df)


def random_split(
    df: pd.DataFrame, train_frac: float, validation_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame into training, validation, and test sets.

    This function performs a random split of the input DataFrame into three
    separate sets. The data is first shuffled to ensure random distribution
    across splits, then divided based on the specified fractions.

    Args:
        df: The input DataFrame to be split.
        train_frac: The fraction of data to allocate to the training set.
            Should be between 0 and 1 (e.g., 0.7 for 70%).
        validation_frac: The fraction of data to allocate to the validation
            set. Should be between 0 and 1 (e.g., 0.15 for 15%). The
            remaining data (1 - train_frac - validation_frac) will be used for
            the test set.

    Returns:
        A tuple containing three DataFrames in order:
            - train_df: Training set DataFrame
            - validation_df: Validation set DataFrame
            - test_df: Test set DataFrame

    Note:
        This function uses a fixed random state (123) for reproducibility.
        The function is adapted from the LLMs-from-scratch repository.

    Example:
        ```python
        train_df, val_df, test_df = random_split(
            balanced_df, train_frac=0.7, validation_frac=0.15
        )
        # Results in 70% train, 15% validation, 15% test split
        ```
    """
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return (
        pd.DataFrame(train_df),
        pd.DataFrame(validation_df),
        pd.DataFrame(test_df),
    )


def download_model(model_size: str = '124M', models_dir: str = 'models'):
    """Downloads GPT-2 model files from OpenAI's blob storage.

    This function fetches the specified GPT-2 model files from OpenAI's public
    blob storage and saves them to a local directory. It displays progress bars
    for each file being downloaded to provide user feedback.

    Args:
        model_size: The size variant of the GPT-2 model to download.
            Must be one of '124M', '355M', '774M', or '1558M'.
            Defaults to '124M' (the smallest model).
        models_dir: The base directory where model files will be stored.
            A subdirectory named after the model_size will be created.
            Defaults to 'models'.

    Returns:
        None. Model files are downloaded to the specified directory structure.

    Raises:
        ValueError: If model_size is not one of the allowed values.

    Note:
        This function is adapted from OpenAI's official GPT-2 repository.
        The downloaded files include checkpoints, tokenizer files, and
        model hyperparameters necessary for loading the pretrained model.

    Example:
        ```python
        # Download the default 124M parameter model
        download_model()

        # Download a larger model to a custom directory
        download_model(model_size='355M', models_dir='pretrained_models')
        ```
    """
    allowed_sizes = ('124M', '355M', '774M', '1558M')
    if model_size not in allowed_sizes:
        raise ValueError(f'Model size not in {allowed_sizes}')

    subdir = os.path.join(models_dir, model_size)
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace('\\', '/')  # needed for Windows

    for filename in [
        'checkpoint',
        'encoder.json',
        'hparams.json',
        'model.ckpt.data-00000-of-00001',
        'model.ckpt.index',
        'model.ckpt.meta',
        'vocab.bpe',
    ]:
        r = requests.get(
            'https://openaipublic.blob.core.windows.net/gpt-2/'
            + subdir
            + '/'
            + filename,
            stream=True,
        )

        with open(os.path.join(subdir, filename), 'wb') as f:
            file_size = int(r.headers['content-length'])
            chunk_size = 1000
            with tqdm(
                ncols=100,
                desc='Fetching ' + filename,
                total=file_size,
                unit_scale=True,
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)


def load_gpt2_params_from_tf_ckpt(
    ckpt_path: str, settings: dict[str, Any]
) -> dict[str, Any]:
    """Loads GPT-2 parameters from a TensorFlow checkpoint file.

    This function reads a TensorFlow checkpoint file containing GPT-2 model
    weights and organizes them into a nested dictionary structure that mirrors
    the model architecture. It handles the complex naming scheme used in the
    original GPT-2 implementation.

    Args:
        ckpt_path: The file path to the TensorFlow checkpoint file
            (e.g., 'models/124M/model.ckpt'). This should be the base path
            without file extensions.
        settings: A dictionary containing the model's configuration, including
            'n_layer' which specifies the number of transformer blocks.

    Returns:
        A nested dictionary containing the model parameters as NumPy arrays.
        The structure includes:
            - 'blocks': List of parameter dictionaries for each transformer
                layer
            - Top-level parameters for embeddings and layer normalization

    Note:
        This function removes singleton dimensions from loaded variables and
        organizes them according to the hierarchical structure of the GPT-2
        architecture.

    Example:
        ```python
        settings = {'n_layer': 12}  # For GPT-2 124M
        params = load_gpt2_params_from_tf_ckpt(
            'models/124M/model.ckpt', settings
        )
        print(f'Loaded {len(params["blocks"])} transformer blocks')
        ```
    """
    # Initialize parameters dictionary with empty blocks for each layer
    params = {'blocks': [{} for _ in range(settings['n_layer'])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split('/')[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict: dict[str, Any] = params
        if variable_name_parts[0].startswith('h'):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params['blocks'][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def load_weights_into_gpt(gpt_model: GPT, params: dict[str, Any]):
    """Loads pre-trained GPT-2 weights into a PyTorch model instance.

    This function transfers weights from a TensorFlow GPT-2 checkpoint (loaded
    as NumPy arrays) into a PyTorch GPT model. It handles the necessary tensor
    conversions, weight matrix transpositions, and the splitting of combined
    weight matrices for multi-head attention.

    Args:
        gpt_model: The PyTorch GPT model instance into which weights will be
            loaded. Must be an instance of the GPT class with the expected
            architecture (transformer blocks, embeddings, etc.).
        params: A nested dictionary containing model parameters as NumPy
            arrays, typically obtained from load_gpt2_params_from_tf_ckpt().

    Returns:
        The same GPT model instance with loaded weights (modified in-place).

    Note:
        This function implements weight tying, where the token embedding matrix
        is shared with the output projection layer. It uses strict=False when
        loading the state dict to accommodate attention mask parameters that
        may not be present in the checkpoint.

    Example:
        ```python
        # Load checkpoint parameters
        params = load_gpt2_params_from_tf_ckpt(ckpt_path, settings)

        # Create model and load weights
        model = GPT(model_config)
        model = load_weights_into_gpt(model, params)
        print('Pre-trained weights loaded successfully')
        ```
    """
    state_dict = {}
    # --- Token and Positional Embeddings ---
    state_dict['tok_embed.weight'] = torch.from_numpy(params['wte'])
    state_dict['pos_embed.weight'] = torch.from_numpy(params['wpe'])

    # --- Transformer Blocks ---
    for b in range(len(params['blocks'])):
        block_params = params['blocks'][b]

        # MultiHeadAttention
        q_w, k_w, v_w = np.split(
            block_params['attn']['c_attn']['w'], 3, axis=-1
        )
        q_b, k_b, v_b = np.split(
            block_params['attn']['c_attn']['b'], 3, axis=-1
        )

        state_dict[f'transformer_blocks.{b}.attn.W_query.weight'] = (
            torch.from_numpy(q_w)
        )
        state_dict[f'transformer_blocks.{b}.attn.W_key.weight'] = (
            torch.from_numpy(k_w)
        )
        state_dict[f'transformer_blocks.{b}.attn.W_value.weight'] = (
            torch.from_numpy(v_w)
        )
        state_dict[f'transformer_blocks.{b}.attn.W_query.bias'] = (
            torch.from_numpy(q_b)
        )
        state_dict[f'transformer_blocks.{b}.attn.W_key.bias'] = (
            torch.from_numpy(k_b)
        )
        state_dict[f'transformer_blocks.{b}.attn.W_value.bias'] = (
            torch.from_numpy(v_b)
        )

        state_dict[f'transformer_blocks.{b}.attn.out_proj.weight'] = (
            torch.from_numpy(block_params['attn']['c_proj']['w'])
        )
        state_dict[f'transformer_blocks.{b}.attn.out_proj.bias'] = (
            torch.from_numpy(block_params['attn']['c_proj']['b'])
        )

        # Feed-Forward Network
        state_dict[f'transformer_blocks.{b}.ff.layers.0.weight'] = (
            torch.from_numpy(block_params['mlp']['c_fc']['w'].T)
        )
        state_dict[f'transformer_blocks.{b}.ff.layers.0.bias'] = (
            torch.from_numpy(block_params['mlp']['c_fc']['b'])
        )
        state_dict[f'transformer_blocks.{b}.ff.layers.2.weight'] = (
            torch.from_numpy(block_params['mlp']['c_proj']['w'].T)
        )
        state_dict[f'transformer_blocks.{b}.ff.layers.2.bias'] = (
            torch.from_numpy(block_params['mlp']['c_proj']['b'])
        )

        # Layer Normalization
        state_dict[f'transformer_blocks.{b}.norm1.weight'] = torch.from_numpy(
            block_params['ln_1']['g']
        )
        state_dict[f'transformer_blocks.{b}.norm1.bias'] = torch.from_numpy(
            block_params['ln_1']['b']
        )
        state_dict[f'transformer_blocks.{b}.norm2.weight'] = torch.from_numpy(
            block_params['ln_2']['g']
        )
        state_dict[f'transformer_blocks.{b}.norm2.bias'] = torch.from_numpy(
            block_params['ln_2']['b']
        )

    # --- Final Layer Normalization and Output Head ---
    state_dict['final_norm.weight'] = torch.from_numpy(params['g'])
    state_dict['final_norm.bias'] = torch.from_numpy(params['b'])

    # Weight tying
    # Weight tying in GPT models is a technique where the embedding layer
    # (converting token IDs to vectors) and the projection layer
    # (converting vectors back to token probabilities) share the same weight
    # matrix.
    # The idea is tha these layers are essentially inverses of each other.
    # Ref: https://wandb.ai/training-transformers-vast/gpt2-sai/reports/Notes-on-Implementing-GPT-2-from-scratch---VmlldzoxMjE4Nzg4NA#weight-tying
    state_dict['out_head.weight'] = torch.from_numpy(params['wte'])

    # Load the created state_dict into the model
    # strict = False for the attention mask
    gpt_model.load_state_dict(state_dict, strict=False)

    return gpt_model


def prepare_classification_model(
    model, model_config, pretrained_model_dir, device
):
    """Prepares a GPT model for classification fine-tuning.

    This function loads pre-trained GPT-2 weights into a PyTorch model,
    freezes most parameters, and adapts the model for binary classification
    by replacing the language modeling head with a classification head.
    It implements a parameter-efficient fine-tuning strategy by only making
    the final transformer block and output layer trainable.

    Args:
        model: The PyTorch GPT model instance to be prepared for
            classification. Should be an instance of the GPT class with the
            standard architecture.
        model_config: Configuration object containing model parameters,
            specifically embed_dim for determining the classification head
            size.
        pretrained_model_dir: Directory path containing the pre-trained GPT-2
            checkpoint files (hparams.json and TensorFlow checkpoint files).
        device: PyTorch device (CPU or CUDA) where the model will be moved
            for training/inference.

    Returns:
        None. The model is modified in-place with the following changes:
            - Pre-trained weights loaded from checkpoint
            - Most parameters frozen (requires_grad=False)
            - Output head replaced with binary classification layer
            - Final transformer block made trainable
            - Final layer normalization made trainable
            - Model moved to specified device

    Note:
        This function implements a common fine-tuning strategy where only
        the top layers are adapted for the downstream task, which helps
        prevent overfitting and reduces computational requirements.

    Example:
        ```python
        model = GPT(model_config)
        prepare_classification_model(
            model=model,
            model_config=model_config,
            pretrained_model_dir='models/124M/',
            device=torch.device('cuda'),
        )
        print(
            f'Model prepared with {
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            } trainable parameters'
        )
        ```
    """
    # load tf model
    tf_ckpt_path = tf.train.latest_checkpoint(pretrained_model_dir)

    with open(
        os.path.join(pretrained_model_dir, 'hparams.json'), encoding='utf-8'
    ) as f:
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
