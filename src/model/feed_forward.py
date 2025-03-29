"""
Feed-forward network implementation for the Grain de Saga model.

This module implements the standard feed-forward network used in transformer models,
which processes each position independently and allows the model to introduce
non-linearities and transform representations between attention layers.
"""

import mlx.core as mx
import mlx.nn as nn

from ..utils.config import ModelConfig


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.

    The feed-forward network consists of two linear transformations with a GELU
    activation in between. It processes each position in the sequence independently,
    allowing the model to transform the attention output into more complex representations.

    In transformer architectures, this network follows the multi-head attention layer
    and introduces non-linearity into the model.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the feed-forward network.

        Args:
            config: Model configuration containing embedding dimension, feed-forward dimension,
                   and dropout rate.
        """
        super().__init__()

        # First linear transformation expands the embedding dimension to the feed-forward dimension.
        # This expansion allows for a more expressive transformation of the input.
        # Typically, the feed-forward dimension is 4x the embedding dimension, though we use a
        # smaller multiplier (3x) to reduce the parameter count for our tiny model.
        self.first_projection = nn.Linear(config.embedding_dimension, config.feed_forward_dimension)

        # Second linear transformation projects back to the original embedding dimension.
        # This ensures the output can be added to the residual connection in the transformer block.
        # The combination of expansion followed by projection creates a bottleneck architecture
        # that helps the model learn more complex patterns while controlling parameter count.
        self.second_projection = nn.Linear(config.feed_forward_dimension, config.embedding_dimension)

        # Dropout for regularization, applied after each transformation.
        # This helps prevent overfitting by randomly setting some activations to zero during training.
        # The probability of dropping a neuron is specified by the dropout_rate in the config.
        self.dropout = nn.Dropout(config.dropout_rate)

        # GELU (Gaussian Error Linear Unit) activation function.
        # GELU was introduced in the paper "Gaussian Error Linear Units (GELUs)"
        # by Hendrycks and Gimpel (2016) and has become a standard in transformer models.
        #
        # Unlike the traditional ReLU (Rectified Linear Unit) which has a hard cutoff at zero,
        # GELU multiplies the input by the cumulative distribution function of the standard
        # normal distribution. This creates a smooth curve that approaches 0 for negative inputs
        # and approaches x for positive inputs, but with a smooth transition.
        #
        # The formula for GELU is approximately:
        # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        #
        # GELU has been shown to outperform ReLU in transformer models and is used in
        # architectures like BERT, GPT, and others.
        self.gelu = nn.GELU()

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Apply the feed-forward network to the input hidden states.

        The feed-forward network processes each position independently, allowing
        the model to introduce non-linearities and transform the attention output.

        Args:
            hidden_states: Input tensor of shape [batch_size, sequence_length, embedding_dimension].

        Returns:
            Transformed hidden states of shape [batch_size, sequence_length, embedding_dimension].
        """
        # First linear projection expands the dimension.
        # This increases the model's capacity to represent complex patterns.
        # The expanded representation gives the model more "space" to perform transformations.
        hidden_states = self.first_projection(hidden_states)

        # Apply GELU activation function.
        # This introduces non-linearity, allowing the model to learn more complex functions.
        # Non-linearities are crucial in neural networks as they enable the approximation
        # of complex functions that couldn't be represented by linear transformations alone.
        # GELU specifically provides a smooth non-linearity that works well with the
        # gradient-based optimization used in training transformer models.
        hidden_states = self.gelu(hidden_states)

        # Apply dropout for regularization.
        # During training, this randomly sets a fraction of the activations to zero,
        # which prevents co-adaptation of neurons and improves generalization.
        # During inference, dropout is typically disabled, and all neurons are active.
        hidden_states = self.dropout(hidden_states)

        # Second linear projection reduces back to original dimension.
        # This ensures the output can be added to the residual connection.
        # The projection compresses the information from the expanded representation
        # back into the model's standard embedding dimension.
        hidden_states = self.second_projection(hidden_states)

        # Apply dropout again for additional regularization.
        # Having dropout after each transformation is a common practice in transformer models.
        # This second application of dropout further reduces overfitting risk by
        # preventing the model from relying too heavily on specific features.
        return self.dropout(hidden_states)
