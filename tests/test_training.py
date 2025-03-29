"""
Tests for the Grain de Saga training components.
"""

import pytest
import mlx.core as mx
import os
import tempfile

from src.model.grain_de_saga import GrainDeSaga
from src.data.dataset import StoryDataset
from src.data.tokenizer import SimpleTokenizer
from src.training.trainer import Trainer
from src.utils.config import ModelConfig, TrainingConfig


@pytest.fixture
def sample_texts():
    """Create sample texts for testing."""
    return [
        "Once upon a time, there was a little girl named Alice.",
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit."
    ] * 5  # Duplicate to have more data


@pytest.fixture
def model_config():
    """Create a model configuration for testing."""
    return ModelConfig(
        vocab_size=100,
        context_length=16,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=64,
        dropout_rate=0.0  # Use 0.0 for deterministic testing
    )


@pytest.fixture
def training_config():
    """Create a training configuration for testing."""
    return TrainingConfig(
        batch_size=2,
        learning_rate=1e-3,
        max_epochs=1,
        checkpoint_dir="test_checkpoints"
    )


@pytest.fixture
def dataset(sample_texts):
    """Create a dataset for testing."""
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.train(sample_texts)

    dataset = StoryDataset(tokenizer, max_length=16, stride=8)
    dataset.load_texts(sample_texts)

    return dataset


@pytest.fixture
def model(model_config):
    """Create a model for testing."""
    return GrainDeSaga(model_config)


def test_trainer_init(model, dataset, training_config):
    """Test trainer initialization."""
    trainer = Trainer(model, dataset, training_config)

    assert trainer.model is model
    assert trainer.dataset is dataset
    assert trainer.config is training_config
    assert trainer.global_step == 0
    assert trainer.best_loss == float('inf')


def test_compute_loss(model, dataset, training_config):
    """Test loss computation."""
    trainer = Trainer(model, dataset, training_config)

    # Get a batch
    inputs, targets = dataset.get_batch(batch_size=2)

    # Compute loss
    loss, logits = trainer.compute_loss(model.parameters(), inputs, targets)

    # Check loss and logits
    assert isinstance(loss, mx.array)
    assert logits.shape[0] == inputs.shape[0]
    assert logits.shape[1] == inputs.shape[1]
    assert logits.shape[2] == model_config().vocab_size


def test_train_step(model, dataset, training_config):
    """Test a single training step."""
    trainer = Trainer(model, dataset, training_
