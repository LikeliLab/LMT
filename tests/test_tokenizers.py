"""Unit tests for tokenizers."""

from lmt.tokenizer.bpe import BPETokenizer
from lmt.tokenizer.naive import NaiveTokenizer


class TestBPETokenizer:
    """Test suite for the BPE tokenizer."""

    def test_bpe_tokenizer_initialization(self):
        """Test BPE tokenizer initializes correctly."""
        tokenizer = BPETokenizer()
        assert hasattr(tokenizer, 'tokenizer')
        assert hasattr(tokenizer, 'allowed_special')
        assert tokenizer.allowed_special == set()

    def test_bpe_tokenizer_initialization_with_params(self):
        """Test BPE tokenizer initializes with custom parameters."""
        allowed_special = {'<|endoftext|>'}
        tokenizer = BPETokenizer(
            encoding_name='gpt2', allowed_special=allowed_special
        )
        assert tokenizer.allowed_special == allowed_special

    def test_bpe_encode_decode_roundtrip(self):
        """Test that encoding then decoding returns original text."""
        tokenizer = BPETokenizer()
        original_text = 'Hello, world! This is a test.'

        encoded = tokenizer.encode(original_text)
        decoded = tokenizer.decode(encoded)

        assert decoded == original_text

    def test_bpe_encode_returns_list_of_ints(self):
        """Test that encode returns a list of integers."""
        tokenizer = BPETokenizer()
        text = 'Hello, world!'

        encoded = tokenizer.encode(text)

        assert isinstance(encoded, list)
        assert all(isinstance(token_id, int) for token_id in encoded)
        assert len(encoded) > 0

    def test_bpe_decode_returns_string(self):
        """Test that decode returns a string."""
        tokenizer = BPETokenizer()
        token_ids = [15496, 11, 995, 0]  # Some example token IDs

        decoded = tokenizer.decode(token_ids)

        assert isinstance(decoded, str)

    def test_bpe_empty_text(self):
        """Test BPE tokenizer with empty text."""
        tokenizer = BPETokenizer()

        encoded = tokenizer.encode('')
        decoded = tokenizer.decode([])

        assert encoded == []
        assert decoded == ''

    def test_bpe_special_tokens(self):
        """Test BPE tokenizer with special tokens."""
        allowed_special = {'<|endoftext|>'}
        tokenizer = BPETokenizer(allowed_special=allowed_special)

        text = 'Hello <|endoftext|> world'
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert '<|endoftext|>' in decoded
        assert isinstance(encoded, list)
        assert len(encoded) > 0

    def test_bpe_different_encodings(self):
        """Test BPE tokenizer with different encoding names."""
        for encoding_name in ['gpt2', 'p50k_base', 'cl100k_base']:
            try:
                tokenizer = BPETokenizer(encoding_name=encoding_name)
                text = 'Hello, world!'
                encoded = tokenizer.encode(text)
                decoded = tokenizer.decode(encoded)
                assert decoded == text
            except ValueError:
                # Some encodings might not be available
                continue


class TestNaiveTokenizer:
    """Test suite for the Naive tokenizer."""

    def test_naive_tokenizer_initialization(self):
        """Test Naive tokenizer initializes correctly."""
        vocab = {'hello': 0, 'world': 1, ',': 2, '<unk>': 3}
        tokenizer = NaiveTokenizer(vocab)

        assert tokenizer.str_to_int == vocab
        assert tokenizer.int_to_str == {
            0: 'hello',
            1: 'world',
            2: ',',
            3: '<unk>',
        }
        assert tokenizer.unknown_str == '<unk>'
        assert tokenizer.unknown_token == 3

    def test_naive_tokenizer_custom_unknown(self):
        """Test Naive tokenizer with custom unknown token."""
        vocab = {'hello': 0, 'world': 1, '<UNK>': 2}
        tokenizer = NaiveTokenizer(vocab, unknown_str='<UNK>')

        assert tokenizer.unknown_str == '<UNK>'
        assert tokenizer.unknown_token == 2

    def test_naive_encode_known_tokens(self):
        """Test encoding with tokens that exist in vocabulary."""
        vocab = {'hello': 0, 'world': 1, ',': 2, '!': 3, '<unk>': 4}
        tokenizer = NaiveTokenizer(vocab)

        encoded = tokenizer.encode('hello, world!')
        expected = [0, 2, 1, 3]  # hello, ,, world, !

        assert encoded == expected

    def test_naive_encode_unknown_tokens(self):
        """Test encoding with tokens not in vocabulary."""
        vocab = {'hello': 0, 'world': 1, '<unk>': 2}
        tokenizer = NaiveTokenizer(vocab)

        encoded = tokenizer.encode('hello unknown')
        expected = [0, 2]  # hello, <unk>

        assert encoded == expected

    def test_naive_decode_known_tokens(self):
        """Test decoding with valid token IDs."""
        vocab = {'hello': 0, 'world': 1, ',': 2, '!': 3, '<unk>': 4}
        tokenizer = NaiveTokenizer(vocab)

        decoded = tokenizer.decode([0, 2, 1, 3])
        expected = 'hello, world!'

        assert decoded == expected

    def test_naive_decode_unknown_tokens(self):
        """Test decoding with invalid token IDs."""
        vocab = {'hello': 0, 'world': 1, '<unk>': 2}
        tokenizer = NaiveTokenizer(vocab)

        decoded = tokenizer.decode([0, 999, 1])  # 999 is not in vocab

        assert '<unk>' in decoded
        assert 'hello' in decoded
        assert 'world' in decoded

    def test_naive_encode_decode_roundtrip(self):
        """Test that encoding then decoding preserves meaning (not necessarily exact text)."""
        vocab = {
            'hello': 0,
            'world': 1,
            ',': 2,
            '!': 3,
            'this': 4,
            'is': 5,
            'a': 6,
            'test': 7,
            '.': 8,
            '<unk>': 9,
        }
        tokenizer = NaiveTokenizer(vocab)

        original_text = 'hello, world! this is a test.'
        encoded = tokenizer.encode(original_text)
        decoded = tokenizer.decode(encoded)

        # Check that all known words are preserved
        for word in ['hello', 'world', 'this', 'is', 'a', 'test']:
            assert word in decoded

    def test_naive_empty_text(self):
        """Test Naive tokenizer with empty text."""
        vocab = {'hello': 0, '<unk>': 1}
        tokenizer = NaiveTokenizer(vocab)

        encoded = tokenizer.encode('')
        decoded = tokenizer.decode([])

        assert encoded == []
        assert decoded == ''

    def test_naive_whitespace_handling(self):
        """Test that whitespace is handled correctly."""
        vocab = {'hello': 0, 'world': 1, '<unk>': 2}
        tokenizer = NaiveTokenizer(vocab)

        # Test with extra whitespace
        text = '  hello    world  '
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert 'hello' in decoded
        assert 'world' in decoded
        # Extra whitespace should be normalized

    def test_naive_special_characters(self):
        """Test Naive tokenizer with special characters."""
        vocab = {
            'hello': 0,
            'world': 1,
            '(': 2,
            ')': 3,
            "'": 4,
            '--': 5,
            ':': 6,
            ';': 7,
            '<unk>': 8,
        }
        tokenizer = NaiveTokenizer(vocab)

        text = "hello (world) it's--good: yes; no"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        # Check that special characters are properly tokenized
        assert 'hello' in decoded
        assert 'world' in decoded
        assert '(' in decoded
        assert ')' in decoded

    def test_naive_vocab_mapping_consistency(self):
        """Test that str_to_int and int_to_str mappings are consistent."""
        vocab = {'apple': 0, 'banana': 1, 'cherry': 2, '<unk>': 3}
        tokenizer = NaiveTokenizer(vocab)

        # Test forward and backward mapping
        for string, int_id in tokenizer.str_to_int.items():
            if int_id in tokenizer.int_to_str:
                assert tokenizer.int_to_str[int_id] == string

        for int_id, string in tokenizer.int_to_str.items():
            if string in tokenizer.str_to_int:
                assert tokenizer.str_to_int[string] == int_id
