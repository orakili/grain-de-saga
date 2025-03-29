"""
Tests for the Grain de Saga model components.
"""

import pytest
import mlx.core as mx

from src.model.embedding import EmbeddingLayer
from src.model.attention import MultiHeadAttention
from src.model.feed_forward import FeedForward
from src.model.layer_norm import RMSNorm
from src.model.transformer import TransformerBlock
from src.model.grain_de_saga import GrainDeSaga
from src.utils.config import ModelConfig


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


def test_embedding_layer(model_config):
    """Test the embedding layer."""
    batch_size, seq_len = 2, 8
    embedding = EmbeddingLayer(model_config)

    # Create input tensor
    input_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)

    # Forward pass
    output = embedding(input_ids)

    # Check output shape
    assert output.shape == (batch_size, seq_len, model_config.embedding_dim)


def test_attention_mechanism(model_config):
    """Test the attention mechanism."""
    batch_size, seq_len = 2, 8
    attention = MultiHeadAttention(model_config)

    # Create input tensor
    x = mx.zeros((batch_size, seq_len, model_config.embedding_dim))

    # Forward pass
    output = attention(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, model_config.embedding_dim)


def test_feed_forward_network(model_config):
    """Test the feed-forward network."""
    batch_size, seq_len = 2, 8
    feed_forward = FeedForward(model_config)

    # Create input tensor
    x = mx.zeros((batch_size, seq_len, model_config.embedding_dim))

    # Forward pass
    output = feed_forward(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, model_config.embedding_dim)


def test_layer_norm(model_config):
    """Test the layer normalization."""
    batch_size, seq_len = 2, 8
    layer_norm = RMSNorm(model_config.embedding_dim)

    # Create input tensor
    x = mx.ones((batch_size, seq_len, model_config.embedding_dim))

    # Forward pass
    output = layer_norm(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, model_config.embedding_dim)


def test_transformer_block(model_config):
    """Test the transformer block."""
    batch_size, seq_len = 2, 8
    transformer_block = TransformerBlock(model_config)

    # Create input tensor
    x = mx.zeros((batch_size, seq_len, model_config.embedding_dim))

    # Forward pass
    output = transformer_block(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, model_config.embedding_dim)


def test_grain_de_saga_model(model_config):
    """Test the full Grain de Saga model."""
    batch_size, seq_len = 2, 8
    model = GrainDeSaga(model_config)

    # Create input tensor
    input_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)

    # Forward pass
    output = model(input_ids)

    # Check output shape
    assert output.shape == (batch_size, seq_len, model_config.vocab_size)


def test_model_generation(model_config):
    """Test the model's generation capability."""
    batch_size, seq_len = 1, 4
    model = GrainDeSaga(model_config)

    # Create input tensor
    input_ids = mx.zeros((batch_size, seq_len), dtype=mx.int32)

    # Generate text
    output = model.generate(input_ids, max_length=8)

    # Check output shape (should be longer than input)
    assert output.shape[1] > seq_len
    assert output.shape[1] <= 8  # Should not exceed max_length
