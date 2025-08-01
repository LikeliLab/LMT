"""Constructs a PyTorch Dataset for GPT model training.

This script defines a custom PyTorch Dataset class, `GPTDataset`, which takes
a raw text corpus and processes it into input-target pairs suitable for
training an autoregressive language model like GPT.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset

from lmt.tokenizer import BaseTokenizer


class GPTDataset(Dataset):
    """A PyTorch Dataset for next-token prediction.

    This dataset processes a single large text corpus into chunks of a
    specified length. For each chunk of text, it creates an input sequence and
    a target sequence. The target sequence is the input sequence shifted by
    one token to the right, which is a standard format for training
    autoregressive language models.

    Attributes:
        input_ids (list[list[int]]): A list of tokenized input sequences.
        target_ids (list[list[int]]): A list of tokenized target sequences.
    """

    def __init__(self, txt, tokenizer, max_length, stride):
        """Initializes the dataset.

        Args:
            txt (str): The raw text corpus to process.
            tokenizer: An object with an `encode` method that converts text
                strings into a list of token IDs.
            max_length (int): The length of each input and target sequence.
            stride (int): The step size to use when creating overlapping
                chunks from the tokenized text. A smaller stride results in
                more overlap and a larger dataset.
        """
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """Returns the total number of samples in the dataset.

        Returns:
            int: The total number of input/target pairs.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """Retrieves a sample from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input
                tensor and the target tensor.
        """
        return self.input_ids[idx], self.target_ids[idx]


class ClassificationDataset(Dataset):
    """A PyTorch Dataset for text classification tasks.

    This class reads a CSV file, tokenizes the text data, and prepares it
    for use in a PyTorch model. It handles tokenization, sequence padding,
    and truncation.

    Attributes:
        data (pd.DataFrame): The DataFrame loaded from the CSV file.
        encoded_texts (List[List[int]]): A list of tokenized and padded
            text sequences.
        max_length (int): The maximum length of the sequences.
    """

    def __init__(
        self,
        csv_file: str,
        tokenizer: BaseTokenizer,
        max_length: int | None = None,
        pad_token_id: int = 50256,
    ):
        """Initializes the ClassificationDataset.

        Args:
            csv_file (str): The path to the input CSV file. The file must
                contain "Text" and "Label" columns.
            tokenizer (Any): A tokenizer object with an `encode` method that
                converts text into a list of token IDs.
            max_length (Optional[int], optional): The maximum sequence length.
                If None, it's set to the length of the longest sequence in the
                dataset. Defaults to None.
            pad_token_id (int, optional): The ID of the token to use for
                padding. Defaults to 50256.
        """
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['Text']
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[: self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_txt + [pad_token_id] * (self.max_length - len(encoded_txt))
            for encoded_txt in self.encoded_texts
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves an item from the dataset at a specific index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the encoded
            text tensor and the corresponding label tensor.
        """
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]['Label']
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self) -> int:
        """Returns the total number of items in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.data)

    def _longest_encoded_length(self) -> int:
        """Calculates the length of the longest tokenized sequence.

        Returns:
            int: The maximum length found among the encoded texts.
        """
        return max(len(encoded_text) for encoded_text in self.encoded_texts)
