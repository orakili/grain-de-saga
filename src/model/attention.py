"""
Attention mechanism implementation for the Grain de Saga model.

This module implements a basic multi-head self-attention mechanism, which is a core
component of transformer models that allows them to focus on different parts of the
input sequence when making predictions.
"""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..utils.config import ModelConfig


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Multi-head attention allows the model to jointly attend to information from
    different representation subspaces at different positions. This enables the model
    to capture various aspects of the input, such as syntactic and semantic relationships.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the multi-head attention module.

        Args:
            config: Model configuration containing embedding dimension, number of heads,
                   and dropout rate.
        """
        super().__init__()

        # Store the embedding dimension for later use.
        self.embedding_dimension = config.embedding_dimension

        # Calculate the dimension of each attention head.
        # The total embedding dimension is split equally among all heads.
        self.head_dimension = config.embedding_dimension // config.num_heads

        # Store the number of attention heads.
        self.num_heads = config.num_heads

        # Create projection matrices for queries, keys, and values.
        # These linear transformations project the input embeddings into
        # separate representation spaces for attention computation.
        self.query_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.key_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension)
        self.value_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension)

        # Create output projection matrix.
        # This linear transformation combines the attention outputs from all heads
        # back into a single representation.
        self.output_projection = nn.Linear(self.embedding_dimension, self.embedding_dimension)

        # Dropout for regularization.
        # Applied to attention weights to prevent overfitting.
        self.dropout = nn.Dropout(config.dropout_rate)

        # Scaling factor for attention scores.
        # Dividing by sqrt(head_dimension) helps maintain the variance of
        # attention weights, preventing extremely small gradients during training.
        self.scale = 1.0 / math.sqrt(self.head_dimension)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Apply multi-head self-attention to the input hidden states.

        Args:
            hidden_states: Input tensor of shape [batch_size, sequence_length, embedding_dimension].
            attention_mask: Optional mask to prevent attention to certain positions.
                           Shape: [batch_size, 1, sequence_length, sequence_length] or
                                 [1, 1, sequence_length, sequence_length].

        Returns:
            Output tensor after attention of shape [batch_size, sequence_length, embedding_dimension].
        """
        # Extract batch size and sequence length from input shape.
        batch_size, sequence_length, _ = hidden_states.shape

        # Project inputs to queries, keys, and values.
        # These projections transform the input embeddings into representations
        # suitable for attention computation.
        queries = self.query_projection(hidden_states)
        keys = self.key_projection(hidden_states)
        values = self.value_projection(hidden_states)

        # Reshape for multi-head attention.
        # Split the embedding dimension into multiple heads, allowing each head
        # to focus on different aspects of the input.
        queries = queries.reshape(batch_size, sequence_length, self.num_heads, self.head_dimension)
        keys = keys.reshape(batch_size, sequence_length, self.num_heads, self.head_dimension)
        values = values.reshape(batch_size, sequence_length, self.num_heads, self.head_dimension)

        # Transpose to [batch_size, num_heads, sequence_length, head_dimension].
        # This shape is more efficient for the subsequent matrix multiplication.
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        # Compute attention scores.
        # The dot product between queries and keys measures how much each position
        # should attend to every other position.
        attention_scores = mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * self.scale

        # Apply mask if provided.
        # The mask prevents attention to certain positions, such as padding tokens
        # or future tokens in autoregressive generation.
        if attention_mask is not None:
            # Adding a large negative value to masked positions ensures they receive
            # negligible attention after softmax.
            attention_scores = attention_scores + attention_mask

        # Apply softmax and dropout.
        # Softmax normalizes the scores to create a probability distribution,
        # and dropout adds regularization.
        attention_weights = mx.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values.
        # This weighted sum combines values according to the attention distribution,
        # allowing the model to focus on relevant parts of the input.
        context_layer = mx.matmul(attention_weights, values)

        # Reshape back to original dimensions.
        # First transpose back to [batch_size, sequence_length, num_heads, head_dimension].
        context_layer = context_layer.transpose(0, 2, 1, 3)

        # Then merge the heads back into a single embedding dimension.
        context_layer = context_layer.reshape(batch_size, sequence_length, self.embedding_dimension)

        # Final projection.
        # This linear transformation combines the outputs from all attention heads
        # into the final representation.
        output = self.output_projection(context_layer)

        return output
