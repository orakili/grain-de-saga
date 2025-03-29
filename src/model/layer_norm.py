"""
Layer normalization implementation for the Grain de Saga model.

This module implements RMSNorm (Root Mean Square Layer Normalization) for stabilizing
the network by normalizing the activations within each layer, which helps with
training stability and convergence.
"""

import mlx.core as mx
import mlx.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is a simplified version of LayerNorm that normalizes by the root mean square
    of the activations. It helps stabilize training by ensuring that the activations
    have consistent scale throughout the network, which is particularly important
    for deep transformer models.

    RMSNorm was introduced in the paper "Root Mean Square Layer Normalization"
    by Zhang and Sennrich (2019) as a more efficient alternative to standard LayerNorm.
    """

    def __init__(self, dimension: int, epsilon: float = 1e-5):
        """
        Initialize the RMSNorm layer.

        Args:
            dimension: The dimension to normalize over (typically the embedding dimension).
            epsilon: Small constant for numerical stability to avoid division by zero.
        """
        super().__init__()

        # Store the epsilon value for use during normalization.
        # This small constant prevents division by zero when the RMS is very small.
        self.epsilon = epsilon

        # Create a learnable scale parameter (gamma) for each dimension.
        # Unlike standard LayerNorm, RMSNorm only uses a scale parameter and no bias.
        # The scale parameter allows the model to modulate the normalized values.
        # We initialize it to zeros and add 1 during the forward pass to start with identity mapping.
        self.scale = mx.zeros((dimension,))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """
        Apply RMS normalization to the input hidden states.

        Args:
            hidden_states: Input tensor of shape [..., dimension].

        Returns:
            Normalized tensor of the same shape.
        """
        # Calculate the root mean square (RMS) value across the last dimension.
        # The RMS is the square root of the average of squared values, which gives
        # a measure of the magnitude of the vector.
        #
        # For a vector x = [x₁, x₂, ..., xₙ], the RMS is:
        # RMS(x) = sqrt((x₁² + x₂² + ... + xₙ²) / n)
        #
        # We add epsilon for numerical stability to avoid division by zero.
        rms = mx.sqrt(mx.mean(hidden_states * hidden_states, axis=-1, keepdims=True) + self.epsilon)

        # Normalize the input by dividing by the RMS value.
        # This ensures that the output has a consistent scale regardless of the
        # magnitude of the input, which helps with training stability.
        normalized_states = hidden_states / rms

        # Apply the learnable scale parameter.
        # We add 1 to the scale parameter to initialize with an identity transformation.
        # This helps with training stability, especially in the early stages.
        #
        # The scale parameter allows the model to modulate the strength of normalization
        # for each dimension, giving it flexibility to learn the optimal scaling.
        scaled_states = normalized_states * (1 + self.scale)

        return scaled_states
