"""Functions needed for finetuning."""

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
from tqdm import tqdm

from lmt.models.gpt import GPT


def download_model(model_size: str = '124M', models_dir: str = 'models'):
    """Downloads the GPT-2 model files from OpenAI's blob storage.

    This function fetches the specified GPT-2 model files and saves them
    to a local directory at 'models/<model_size>'. It displays a progress
    bar for each file being downloaded.
    Copied from: https://github.com/openai/gpt-2/blob/master/download_model.py

    Args:
        model_size (str): The name of the GPT-2 model to download.
                          Defaults to '124M'. Must be one of '124M',
                          '355M', '774M', or '1558M'.
        models_dir (str): Directory to save model weights to

    Raises:
        ValueError: If the `model_size` is not one of the allowed values.
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

    This function iterates through variables in a TensorFlow checkpoint,
    parses their names to determine their place in the model architecture,
    and loads them into a nested dictionary of NumPy arrays.

    Args:
        ckpt_path: The file path to the TensorFlow checkpoint
            (e.g., 'models/124M/model.ckpt').
        settings: A dictionary containing the model's configuration settings,
                  such as 'n_layer', which specifies the number of transformer
                blocks.

    Returns:
        A nested dictionary containing the model parameters.
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

    This function iterates through a dictionary of weights (loaded from a
    TensorFlow checkpoint) and assigns them to the corresponding layers and
    parameters of a PyTorch GPT model. It handles the necessary transpositions
    and splitting of combined weight matrices (e.g., for query, key, value).

    Args:
        gpt_model: The PyTorch GPT model instance (an object of a class
            inheriting from `torch.nn.Module`) into which the weights will be
            loaded.
        params: A nested dictionary containing the model parameters as NumPy
                arrays, structured to match the original GPT-2 architecture.

    Returns:
        None. The model is modified in-place.
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


def download_and_unzip_spam_data(
    url: str = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip',
    zip_path: Path = Path('data/sms_spam_collection.zip'),
    extracted_path: Path = Path('sms_spam_collection'),
    data_file_path: Path = Path('data/SMSSpamCollection.tsv'),
) -> None:
    """Downloads, extracts, and renames the SMS Spam Collection dataset.

    Checks if the final data file already exists. If not, it downloads the
    zip archive from the specified URL, extracts its contents, and renames
    the data file to include a .tsv extension.
    Copied from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb

    Args:
        url: The URL to download the zip file from.
        zip_path: The local path to save the downloaded .zip file.
        extracted_path: The directory where the contents of the zip file
            will be extracted.
        data_file_path: The final path for the dataset file, including
            the desired .tsv extension.
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
    """Creates a balanced dataset by downsampling the majority class ('ham').

    This function counts the number of instances of the minority class ('spam')
    and randomly samples the majority class ('ham') to match that count.
    It then concatenates the downsampled 'ham' instances with all 'spam'
    instances to create a new, balanced DataFrame.
    Copied from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb

    Args:
        df: The input DataFrame, which must contain a 'Label' column
            with 'spam' and 'ham' values.

    Returns:
        A new DataFrame with an equal number of 'spam' and 'ham' samples.
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

    This function first shuffles the entire DataFrame randomly. It then splits
    the shuffled data into three separate DataFrames (training, validation,
    and test) based on the provided fractions.
    Copied from: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch06/01_main-chapter-code/ch06.ipynb

    Args:
        df: The input DataFrame to be split.
        train_frac: The fraction of the data to be allocated to the
            training set (e.g., 0.7).
        validation_frac: The fraction of the data to be allocated to the
            validation set (e.g., 0.15). The remaining data will be
            used for the test set.

    Returns:
        A tuple containing three DataFrames:
        (train_df, validation_df, test_df).
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
