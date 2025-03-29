"""
Configuration module for the Grain de Saga language model.

This module defines the configuration classes and default parameters
for the model architecture and training process.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """
    Configuration for the Grain de Saga model architecture.

    This class contains all hyperparameters that define the model's structure,
    including vocabulary size, dimensions, and regularization settings.
    """

    # Vocabulary size determines the number of unique tokens the model can process.
    # A larger vocabulary can represent more nuanced language but increases parameter count.
    vocabulary_size: int = 8_000

    # Context length is the maximum sequence length the model can process at once.
    # This limits how much text the model can "see" when making predictions.
    context_length: int = 512

    # Embedding dimension controls the size of the vector space where tokens are represented.
    # Higher dimensions can capture more semantic information but require more computation.
    embedding_dimension: int = 128

    # Number of transformer layers in the model.
    # More layers allow for more complex representations but increase computation and parameters.
    num_layers: int = 3

    # Number of attention heads in each transformer layer.
    # Multiple heads allow the model to attend to different parts of the input simultaneously.
    num_heads: int = 4

    # Feed-forward dimension is the size of the intermediate representation in the FFN.
    # Typically set to 4x the embedding dimension, but we use 3x to reduce parameters.
    feed_forward_dimension: int = 384

    # Dropout rate controls the amount of regularization applied during training.
    # Higher values help prevent overfitting but may slow down convergence.
    dropout_rate: float = 0.1

    def __post_init__(self):
        """
        Validate configuration parameters after initialization.

        This method ensures that the model configuration is valid and consistent.
        For example, the embedding dimension must be divisible by the number of heads
        for the multi-head attention mechanism to work properly.
        """
        assert self.embedding_dimension % self.num_heads == 0, (
            "Embedding dimension must be divisible by number of heads. "
            f"Got embedding_dimension={self.embedding_dimension} and num_heads={self.num_heads}."
        )


@dataclass
class TrainingConfig:
    """
    Configuration for the training process.

    This class contains hyperparameters related to the optimization process,
    including batch size, learning rate, and training duration.
    """

    # Batch size determines how many sequences are processed in parallel.
    # Larger batches provide more stable gradient estimates but require more memory.
    batch_size: int = 32

    # Learning rate controls the step size during optimization.
    # This value is crucial for convergence - too high can cause divergence,
    # too low can make training extremely slow.
    learning_rate: float = 3e-4

    # Weight decay is an L2 regularization parameter that helps prevent overfitting.
    # It penalizes large weights, encouraging the model to learn simpler patterns.
    weight_decay: float = 0.01

    # Maximum number of training epochs (complete passes through the dataset).
    # More epochs allow for more learning but risk overfitting.
    max_epochs: int = 10

    # Warmup steps gradually increase the learning rate at the beginning of training.
    # This helps stabilize early training, especially with transformer models.
    warmup_steps: int = 100

    # Directory where model checkpoints will be saved during training.
    checkpoint_directory: str = "checkpoints"

    # Interval (in steps) for logging training metrics.
    # This controls how often training progress is reported.
    log_interval: int = 10

    # Random seed for reproducibility.
    # Setting a fixed seed ensures that training runs can be replicated.
    seed: int = 42
