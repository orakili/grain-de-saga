"""
Transformer block implementation for the Grain de Saga model.

This module implements a single transformer block with attention and feed-forward networks,
which forms the core computational unit of the transformer architecture.
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ..utils.config import ModelConfig
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import RMSNorm


class TransformerBlock(nn.Module):
    """
    A single transformer block with attention and feed-forward networks.

    The transformer block is the fundamental building block of transformer-based
    language models. It consists of a multi-head self-attention mechanism followed
    by a position-wise feed-forward network, with residual connections and
    layer normalization applied to both sub-layers.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the transformer block.

        Args:
            config: Model configuration containing dimensions, number of heads,
                   and other hyperparameters.
        """
        super().__init__()

        # Multi-head attention mechanism allows the model to jointly attend to
        # information from different representation subspaces at different positions.
        self.attention = MultiHeadAttention(config)

        # Feed-forward network processes each position independently, allowing the model
        # to transform the attention output into a more complex representation.
        self.feed_forward = FeedForward(config)

        # Layer normalization helps stabilize training by normalizing the inputs
        # to each sub-layer. We use RMSNorm, which is a simplified and more
        # efficient variant of LayerNorm.
        self.layer_norm1 = RMSNorm(config.embedding_dimension)
        self.layer_norm2 = RMSNorm(config.embedding_dimension)

        # Dropout is applied for regularization to prevent overfitting.
        self.dropout = nn.Dropout(config.dropout_rate)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        """
        Apply the transformer block to the input hidden states.

        This implementation uses the Pre-LN (Pre-Layer Normalization) architecture,
        which applies layer normalization before each sub-layer rather than after.
        Pre-LN has been shown to improve training stability, especially for deeper models.

        Args:
            hidden_states: Input tensor of shape [batch_size, sequence_length, embedding_dimension].
            attention_mask: Optional mask to prevent attention to certain positions.
                           Shape: [batch_size, 1, sequence_length, sequence_length] or
                                 [1, 1, sequence_length, sequence_length].

        Returns:
            Transformed hidden states of the same shape as the input.
        """
        # First sub-layer: Multi-head self-attention with residual connection
        # 1. Apply layer normalization to the input (Pre-LN architecture)
        normalized_hidden_states = self.layer_norm1(hidden_states)

        # 2. Apply multi-head self-attention
        # The attention mechanism allows each position to attend to all positions
        # in the previous layer, capturing contextual relationships.
        attention_output = self.attention(normalized_hidden_states, attention_mask)

        # 3. Add residual connection
        # Residual connections help with gradient flow during backpropagation,
        # allowing the model to learn more effectively, especially when deep.
        hidden_states = hidden_states + attention_output

        # Second sub-layer: Position-wise feed-forward network with residual connection
        # 1. Apply layer normalization (Pre-LN architecture)
        normalized_hidden_states = self.layer_norm2(hidden_states)

        # 2. Apply feed-forward network
        # The feed-forward network consists of two linear transformations with a
        # non-linearity in between, applied independently to each position.
        feed_forward_output = self.feed_forward(normalized_hidden_states)

        # 3. Add residual connection
        hidden_states = hidden_states + feed_forward_output

        return hidden_states
