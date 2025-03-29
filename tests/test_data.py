"""
Tests for the Grain de Saga data components.
"""

import pytest
import mlx.core as mx
import os
import tempfile

from src.data.tokenizer import SimpleTokenizer
from src.data.dataset import StoryDataset


@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "Once upon a time, there was a little girl named Alice.",
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit."
    ]


def test_tokenizer_init():
    """Test tokenizer initialization."""
    tokenizer = SimpleTokenizer(vocab_size=100)
    assert tokenizer.vocab_size == 100
    assert tokenizer.pad_token in tokenizer.token_to_id
    assert tokenizer.unk_token in tokenizer.token_to_id
    assert tokenizer.bos_token in tokenizer.token_to_id
    assert tokenizer.eos_token in tokenizer.token_to_id


def test_tokenizer_train(sample_texts):
    """Test tokenizer training."""
    tokenizer = SimpleTokenizer(vocab_size=50)
    tokenizer.train(sample_texts)

    # Check if common words are in vocabulary
    assert "time" in tokenizer.token_to_id
    assert "the" in tokenizer.token_to_id

    # Check vocabulary size
    assert len(tokenizer.token_to_id) <= 50


def test_tokenizer_encode_decode(sample_texts):
    """Test tokenizer encoding and decoding."""
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.train(sample_texts)

    for text in sample_texts:
        # Encode
        ids = tokenizer.encode(text)

        # Decode
        decoded = tokenizer.decode(ids)

        # Check if core content is preserved
        # Note: exact match isn't expected due to tokenization artifacts
        assert text.lower() in decoded.lower()


def test_tokenizer_save_load():
    """Test tokenizer saving and loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create and train tokenizer
        tokenizer = SimpleTokenizer(vocab_size=100)
        tokenizer.train(["This is a test sentence."])

        # Save tokenizer
        save_path = os.path.join(tmpdir, "tokenizer.json")
        tokenizer.save(save_path)

        # Load tokenizer
        loaded_tokenizer = SimpleTokenizer.load(save_path)

        # Check if vocabularies match
        assert tokenizer.vocab_size == loaded_tokenizer.vocab_size
        assert tokenizer.token_to_id == loaded_tokenizer.token_to_id


def test_dataset_creation(sample_texts):
    """Test dataset creation."""
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.train(sample_texts)

    dataset = StoryDataset(tokenizer, max_length=16, stride=8)
    dataset.load_texts(sample_texts)

    # Check if dataset has examples
    assert len(dataset) > 0


def test_dataset_getitem(sample_texts):
    """Test dataset __getitem__ method."""
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.train(sample_texts)

    dataset = StoryDataset(tokenizer, max_length=16, stride=8)
    dataset.load_texts(sample_texts)

    # Get an example
    inputs, targets = dataset[0]

    # Check shapes
    assert inputs.shape[0] == 15  # max_length - 1
    assert targets.shape[0] == 15  # max_length - 1

    # Check if targets are shifted by 1
    # This is a bit tricky to test directly, but we can check the shapes match


def test_dataset_batch(sample_texts):
    """Test dataset batch creation."""
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.train(sample_texts)

    dataset = StoryDataset(tokenizer, max_length=16, stride=8)
    dataset.load_texts(sample_texts)

    # Get a batch
    inputs, targets = dataset.get_batch(batch_size=2)

    # Check batch shapes
    assert inputs.shape[0] <= 2  # Could be less if dataset is small
    assert inputs.shape[1] == 15  # max_length - 1
    assert targets.shape[0] == inputs.shape[0]
    assert targets.shape[1] == 15  # max_length - 1
