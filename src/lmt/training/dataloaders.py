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
"""Pytorch dataloaders needed for training.

This module contains factory functions to create PyTorch DataLoaders for both
language model pretraining and text classification tasks.
"""

from pathlib import Path

from torch.utils.data import DataLoader

from lmt.tokenizer import BaseTokenizer, BPETokenizer
from lmt.training.datasets import ClassificationDataset, GPTDataset


def create_pretraining_dataloader(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer: BaseTokenizer | None = None,
) -> DataLoader:
    """Creates a PyTorch DataLoader for pretraining text data.

    Loads a text file, tokenizes it, and wraps it in a PyTorch DataLoader
    suitable for training a language model. The dataset is configured for
    chunking text into sequences of a specified maximum length with an optional
    stride.

    Args:
        txt (str): Path to the text file containing the pretraining data.
        batch_size (int, optional): Number of samples per batch. Defaults to 4.
        max_length (int, optional): Maximum sequence length for each chunk.
            Defaults to 256.
        stride (int, optional): The number of tokens to "slide" the window for
            chunking. Defaults to 128.
        shuffle (bool, optional): If True, the data is shuffled at each epoch.
            Defaults to True.
        drop_last (bool, optional): If True, drops the last incomplete batch.
            Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. Defaults to 0.
        tokenizer (BaseTokenizer | None, optional): The tokenizer to use. If
            None, a default BPETokenizer is used. Defaults to None.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    if tokenizer is None:
        tokenizer = BPETokenizer('gpt2', allowed_special={'<|endoftext|>'})

    dataset = GPTDataset(txt, tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


def create_classification_dataloader(
    csv_file: Path,
    batch_size: int = 2,
    max_length: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    tokenizer: BaseTokenizer | None = None,
) -> DataLoader:
    """Creates a PyTorch DataLoader for classification data.

    Loads a CSV file with text and labels, tokenizes the text, and wraps it in
    a PyTorch DataLoader for classification tasks.

    Args:
        csv_file (str): Path to the CSV file with the classification data. The
            CSV should contain columns for text and labels.
        batch_size (int, optional): Number of samples per batch. Defaults to 2.
        max_length (int | None, optional): Maximum sequence length for each
            text sample. If None, the tokenizer's default is used.
            Defaults to None.
        shuffle (bool, optional): If True, the data is shuffled at each epoch.
            Defaults to True.
        drop_last (bool, optional): If True, drops the last incomplete batch.
            Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. Defaults to 0.
        tokenizer (BaseTokenizer | None, optional): The tokenizer to use. If
            None, a default BPETokenizer is used. Defaults to None.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    if tokenizer is None:
        tokenizer = BPETokenizer('gpt2', allowed_special={'<|endoftext|>'})

    dataset = ClassificationDataset(
        csv_file=csv_file,
        max_length=max_length,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
