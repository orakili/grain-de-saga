"""
Grain de Saga: A tiny language model for generating children's stories.

This module defines the main GrainDeSaga class, which integrates all components
of the model architecture into a complete language model capable of generating
short stories for children.
"""

from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..utils.config import ModelConfig
from .embedding import EmbeddingLayer
from .layer_norm import RMSNorm
from .transformer import TransformerBlock


class GrainDeSaga(nn.Module):
    """
    The main class for the Grain de Saga language model.

    This class integrates the embedding layer, transformer blocks, and output layer
    to create a complete language model capable of generating children's stories.
    It follows the standard transformer decoder architecture with self-attention
    and feed-forward layers.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the Grain de Saga model.

        Args:
            config: Configuration object containing model hyperparameters such as
                   vocabulary size, embedding dimension, number of layers, etc.
        """
        super().__init__()
        self.config = config

        # Embedding layer converts token IDs to vector representations and adds
        # positional information to preserve sequence order.
        self.embedding = EmbeddingLayer(config)

        # Transformer blocks are the core of the model, each containing
        # self-attention and feed-forward networks.
        # We create a list of transformer blocks based on the specified number of layers.
        self.transformer_blocks = [TransformerBlock(config) for _ in range(config.num_layers)]

        # Final layer normalization is applied before the output projection.
        # This helps stabilize the last layer's outputs and is a standard
        # component in transformer models like GPT.
        self.final_layer_norm = RMSNorm(config.embedding_dimension)

        # Language modeling head projects the final hidden states to vocabulary logits.
        # These logits represent unnormalized probabilities for each token in the vocabulary.
        self.language_model_head = nn.Linear(config.embedding_dimension, config.vocabulary_size)

        # Create causal attention mask to prevent attending to future tokens.
        # In a causal language model, each token can only attend to itself and previous tokens.
        # We create a triangular mask where each position can attend to itself (diagonal)
        # and previous positions (below diagonal), but not to future positions (above diagonal).
        mask = mx.full((config.context_length, config.context_length), float("-inf"))
        mask = mx.triu(mask, k=1)  # Upper triangular part (excluding diagonal) is set to -inf

        # Store the mask as an instance variable.
        # Unlike PyTorch's register_buffer, in MLX we simply store it as an attribute.
        self.attention_mask = mask

    def __call__(
        self,
        input_ids: mx.array,
        return_logits: bool = True
    ) -> mx.array:
        """
        Forward pass of the model.

        Args:
            input_ids: Input tensor of token indices of shape [batch_size, sequence_length].
            return_logits: Whether to return logits (True) or probabilities (False).

        Returns:
            Output logits or probabilities for each token in the vocabulary,
            of shape [batch_size, sequence_length, vocabulary_size].
        """
        batch_size, sequence_length = input_ids.shape

        # Ensure the input sequence length doesn't exceed the maximum context length.
        assert sequence_length <= self.config.context_length, (
            f"Input sequence length {sequence_length} exceeds maximum context length "
            f"{self.config.context_length}."
        )

        # Get the appropriate section of the attention mask for this sequence length.
        # We only need a mask that matches the current sequence length, not the full context length.
        current_attention_mask = self.attention_mask[:sequence_length, :sequence_length]

        # Embed tokens and add positional information.
        # This converts the token IDs to continuous vector representations and
        # adds information about the position of each token in the sequence.
        hidden_states = self.embedding(input_ids)

        # Apply transformer blocks sequentially.
        # Each block contains self-attention and feed-forward networks,
        # allowing the model to process contextual information.
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, current_attention_mask)

        # Apply final layer normalization.
        # This normalizes the output of the last transformer block before
        # projecting to vocabulary logits.
        normalized_states = self.final_layer_norm(hidden_states)

        # Project to vocabulary logits.
        # These logits represent unnormalized probabilities for each token in the vocabulary.
        logits = self.language_model_head(normalized_states)

        if return_logits:
            # Return raw logits for training (used with cross-entropy loss).
            return logits
        else:
            # Convert to probabilities using softmax for generation.
            # Softmax normalizes the logits to create a probability distribution
            # over the vocabulary for each position.
            return mx.softmax(logits, axis=-1)

    def generate(
        self,
        input_ids: mx.array,
        max_length: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> mx.array:
        """
        Generate text from the model using autoregressive sampling.

        This method implements autoregressive generation, where the model generates
        one token at a time and each new token is conditioned on all previous tokens.

        Args:
            input_ids: Input tensor of token indices of shape [batch_size, sequence_length].
                      This serves as the prompt for generation.
            max_length: Maximum length of the generated sequence (including prompt).
            temperature: Sampling temperature (higher = more random, lower = more deterministic).
                        A temperature of 1.0 uses the raw probabilities, while values < 1.0
                        make the distribution more peaked and values > 1.0 make it more uniform.
            top_k: If specified, only sample from the top k most likely tokens.
                  This helps prevent sampling from the long tail of the distribution.

        Returns:
            Generated token indices of shape [batch_size, new_sequence_length].
        """
        batch_size, sequence_length = input_ids.shape
        generated_sequence = input_ids

        # Generate tokens one by one until we reach max_length or context_length.
        for _ in range(max_length - sequence_length):
            # Get predictions for the next token.
            # We only need the logits for the last token in each sequence.
            logits = self(generated_sequence)[:, -1, :]

            # Apply temperature scaling.
            # Higher temperature makes the distribution more uniform,
            # lower temperature makes it more peaked.
            # We use max(temperature, 1e-7) to avoid division by zero.
            logits = logits / max(temperature, 1e-7)

            # Apply top-k sampling if specified.
            # This restricts sampling to only the k most likely tokens,
            # which helps prevent generating low-probability tokens.
            if top_k is not None:
                # Get the values and indices of the top-k logits.
                top_values = mx.topk(logits, min(top_k, logits.shape[-1]))

                # Create a mask where logits below the k-th largest are set to -inf.
                # This effectively removes them from consideration during sampling.
                logits = mx.where(logits < top_values[:, [-1]], float("-inf"), logits)

            # Sample from the distribution using categorical sampling.
            # This converts the logits to probabilities using softmax and then
            # samples from the resulting multinomial distribution.
            probabilities = mx.softmax(logits, axis=-1)
            next_token = mx.random.categorical(probabilities, axis=-1)

            # Append the new token to the generated sequence.
            # We reshape next_token to [batch_size, 1] to match the sequence dimension.
            generated_sequence = mx.concatenate([generated_sequence, next_token[:, None]], axis=1)

            # Break if we exceed the maximum context length.
            # This prevents generating sequences longer than the model can handle.
            if generated_sequence.shape[1] >= self.config.context_length:
                break

        return generated_sequence
