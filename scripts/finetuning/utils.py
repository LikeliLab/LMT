"""Functions needed for finetuning."""

import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
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


def calc_accuracy_loader(
    data_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Calculates the accuracy of a model on a given data loader.

    Args:
        data_loader: The data loader containing the dataset.
        model: The model to evaluate.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.
        num_batches: The number of batches to use for the calculation. If None,
            all batches in the data loader are used.

    Returns:
        The accuracy of the model as a float between 0 and 1.
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return correct_predictions / num_examples


def calc_loss_loader(
    data_loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """Calculates the average loss of a model on a given data loader.

    Args:
        data_loader: The data loader containing the dataset.
        model: The model to evaluate.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.
        num_batches: The number of batches to use for the calculation. If None,
            all batches in the data loader are used.

    Returns:
        The average loss of the model as a float. Returns `nan` if the
        data loader is empty.
    """
    total_loss = 0.0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """Evaluates the model on a subset of the training and validation data.

    Args:
        model: The model to be evaluated.
        train_loader: The data loader for the training set.
        val_loader: The data loader for the validation set.
        device: The device (e.g., 'cpu' or 'cuda') to run the evaluation on.
        eval_iter: The number of batches to use for evaluation from each data
            loader.

    Returns:
        A tuple containing the average training loss and the average validation
        loss, respectively.
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )

        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Calculates the cross-entropy loss for a single batch.

    Args:
        input_batch (torch.Tensor): The input batch of data.
        target_batch (torch.Tensor): The target batch of data.
        model (torch.nn.Module): The neural network model.
        device (str): The device (e.g., 'cpu' or 'cuda') to run the tensors on.

    Returns:
        torch.Tensor: The calculated cross-entropy loss for the batch.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def plot_metrics(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    eval_freq: int,
    save_dir: str,
):
    """Plots the training and validation loss and accuracy over time.

    Args:
        train_losses (list): A list of training loss values.
        val_losses (list): A list of validation loss values.
        train_accs (list): A list of training accuracy values.
        val_accs (list): A list of validation accuracy values.
        eval_freq (int): The frequency (in steps) at which loss was evaluated.
        save_dir (str): Directory to save plot to.
    """
    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Plot the Loss ---
    loss_steps = np.arange(0, len(train_losses) * eval_freq, eval_freq)

    ax1.plot(loss_steps, train_losses, label='Training Loss', marker='o')
    ax1.plot(loss_steps, val_losses, label='Validation Loss', marker='o')

    ax1.set_title('Loss over Training Steps')
    ax1.set_xlabel('Global Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- Plot the Accuracy ---
    # The x-axis for accuracy corresponds to the number of epochs.
    accuracy_epochs = np.arange(1, len(train_accs) + 1)

    ax2.plot(
        accuracy_epochs,
        np.array(train_accs) * 100,
        label='Training Accuracy',
        marker='o',
    )
    ax2.plot(
        accuracy_epochs,
        np.array(val_accs) * 100,
        label='Validation Accuracy',
        marker='o',
    )

    ax2.set_title('Accuracy per Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to prevent titles from overlapping
    plt.tight_layout()

    plt.savefig(f'{save_dir}/training_metrics.png')
    print(f'Plot saved to {save_dir}/training_metrics.png')


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
):
    """Trains a GPT-style model for classification.

    This function handles the complete training loop, including periodic
    evaluation on a validation set and generating sample text to monitor
    the model's progress.

    Args:
        model (torch.nn.Module): The GPT model to be trained.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the
            training set.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the
            validation set.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        device (torch.device): The device to run the model on (e.g., 'cuda'
            or 'cpu').
        num_epochs (int): Total number of epochs to train for.
        eval_freq (int): Frequency (in epochs) to perform evaluation.
        eval_iter (int): Number of batches from val_dataloader to use for
            evaluation.
        start_context (str): A starting string to prompt the model for
            text generation.
        tokenizer: (BaseTokenizer) The tokenizer used for encoding and
            decoding text.
    """
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_dataloader, val_dataloader, device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f'Ep {epoch + 1} (Step {global_step:06d}): '
                    f'Train loss {train_loss:.3f}, Val loss {val_loss:.3f}'
                )

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_dataloader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_dataloader, model, device, num_batches=eval_iter
        )
        print(f'Training accuracy: {train_accuracy * 100:.2f}% | ', end='')
        print(f'Validation accuracy: {val_accuracy * 100:.2f}%')
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen
