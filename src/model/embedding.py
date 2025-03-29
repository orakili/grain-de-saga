"""
Embedding layer implementation for the Grain de Saga model.

This module implements token and position embeddings, which are fundamental
components of transformer-based language models.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..utils.config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Embedding layer for tokens.

    In language models, token embeddings convert discrete token IDs into
    continuous vector representations that capture semantic relationships.
    """

    def __init__(self, vocabulary_size: int, embedding_dimension: int):
        """
        Initialize the token embedding layer.

        Args:
            vocabulary_size: Size of the vocabulary (number of unique tokens).
            embedding_dimension: Dimension of the embedding vectors.
        """
        super().__init__()

        # Create an embedding layer with the specified vocabulary size and
        # embedding dimension.
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        # Initialize the embedding weights with scaled normal distribution.
        # This scaling helps maintain consistent variance across different
        # embedding sizes and aids in stabilizing the forward pass and gradient
        # flow during training.
        self.embedding.weight = mx.random.normal(
            shape=(vocabulary_size, embedding_dimension)
        ) / mx.sqrt(mx.array(embedding_dimension))

    def __call__(self, token_ids: mx.array) -> mx.array:
        """
        Apply token embedding to convert token IDs to vectors.

        Args:
            token_ids: Input tensor of token IDs.

        Returns:
            Embedded token vectors.
        """
        return self.embedding(token_ids)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Positional encodings add information about token positions in the sequence,
    which is crucial because transformer attention has no inherent notion of order.
    Sinusoidal encodings use sine and cosine functions of different frequencies
    to represent position information.
    """

    def __init__(self, embedding_dimension: int, max_sequence_length: int = 512):
        """
        Initialize the positional encoding.

        Args:
            embedding_dimension: Dimension of the embeddings.
            max_sequence_length: Maximum sequence length the model can handle.
        """
        super().__init__()

        # Create a matrix to hold position encodings for all possible positions.
        # This will have shape [max_sequence_length, embedding_dimension].
        positional_encoding = mx.zeros((max_sequence_length, embedding_dimension))

        # Create a column vector of position indices from 0 to max_sequence_length-1.
        # Think of this as the "position" of each token in the sequence.
        position_indices = mx.arange(0, max_sequence_length).reshape(-1, 1)

        # The sinusoidal encoding uses different frequencies for different dimensions.
        # For each dimension i, we use frequency 1/(10000^(i/d_model)).
        # This creates a geometric progression from high frequency to low frequency.
        #
        # Intuition: Think of each dimension as a clock hand moving at different speeds.
        # - The first dimensions are like seconds hands (moving quickly)
        # - The last dimensions are like hour hands (moving slowly)
        # Together, they create a unique pattern for each position.
        division_terms = mx.exp(
            mx.arange(0, embedding_dimension, 2) *
            (-mx.log(mx.array(10000.0)) / embedding_dimension)
        )

        # Apply sine to even indices (0, 2, 4...) in the embedding dimension.
        # This creates wave patterns with different frequencies for each dimension.
        positional_encoding[:, 0::2] = mx.sin(position_indices * division_terms)

        # Apply cosine to odd indices (1, 3, 5...) in the embedding dimension.
        # Using both sine and cosine ensures that each position has a unique encoding.
        # This is important because sin(α+β) can be expressed as a linear combination
        # of sin(α) and cos(α), allowing the model to easily learn relative positions.
        positional_encoding[:, 1::2] = mx.cos(position_indices * division_terms)

        # In MLX, we don't have register_buffer, so we simply store the tensor as an
        # instance variable. This tensor is not a trainable parameter but will be part
        # of the module's state.
        #
        # Unlike PyTorch where register_buffer handles device placement automatically,
        # in MLX we don't need to worry about this as MLX handles device placement
        # differently.
        self.positional_encoding = positional_encoding

    def __call__(self, embeddings: mx.array) -> mx.array:
        """
        Add positional encoding to token embeddings.

        Args:
            embeddings: Token embeddings of shape [batch_size, sequence_length, embedding_dimension].

        Returns:
            Embeddings with positional information added.
        """
        # Extract the actual sequence length from the input.
        sequence_length = embeddings.shape[1]

        # Add the positional encoding to the token embeddings.
        # This simple addition combines semantic information (from token embeddings)
        # with positional information (from positional encoding).
        #
        # Unlike RoPE (Rotary Position Embedding) which uses rotation operations,
        # sinusoidal encoding directly adds position vectors to token embeddings.
        # This additive approach is simpler but may not preserve token information
        # as well as multiplicative approaches like RoPE.
        return embeddings + self.positional_encoding[:sequence_length]

class EmbeddingLayer(nn.Module):
    """
    Combined token and positional embedding layer.

    This layer combines token embeddings (semantic information) with positional
    embeddings (sequence order information) to create the input representation
    for transformer layers.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the embedding layer.

        Args:
            config: Model configuration containing vocabulary size, embedding dimension,
                   context length, and dropout rate.
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(config.vocabulary_size, config.embedding_dimension)
        self.position_embedding = PositionalEncoding(config.embedding_dimension, config.context_length)
        self.dropout = nn.Dropout(config.dropout_rate)

    def __call__(self, token_ids: mx.array) -> mx.array:
        """
        Apply token and positional embeddings.

        Args:
            token_ids: Input tensor of token IDs.

        Returns:
            Combined token and positional embeddings with dropout applied.
        """
        # First convert tokens to embeddings
        embeddings = self.token_embedding(token_ids)

        # Add positional information
        embeddings_with_position = self.position_embedding(embeddings)

        # Apply dropout for regularization
        return self.dropout(embeddings_with_position)
