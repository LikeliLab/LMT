"""Constructs a PyTorch Dataset for GPT model training.

This script defines a custom PyTorch Dataset class, `GPTDataset`, which takes
a raw text corpus and processes it into input-target pairs suitable for
training an autoregressive language model like GPT.
"""

import torch
from torch.utils.data import Dataset


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
