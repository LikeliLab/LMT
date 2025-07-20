"""Abstract base class for tokenizers."""

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the tokenizer with a vocabulary."""
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token IDs."""
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string."""
        pass
