"""
Training module for the Grain de Saga model.

This module provides the training loop and evaluation functionality for the
Grain de Saga language model, handling the optimization process, loss computation,
and model checkpointing.
"""

import time
import os
from typing import Dict, List, Optional, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optimizers
from mlx.utils import tree_flatten

from ..model.grain_de_saga import GrainDeSaga
from ..data.dataset import StoryDataset
from ..utils.config import ModelConfig, TrainingConfig


class Trainer:
    """
    Trainer for the Grain de Saga model.

    This class handles the training process for the language model, including
    optimization, loss computation, evaluation, and checkpointing. It provides
    a high-level interface for training the model on a dataset of stories.
    """

    def __init__(
        self,
        model: GrainDeSaga,
        dataset: StoryDataset,
        config: TrainingConfig
    ):
        """
        Initialize the trainer.

        Args:
            model: The GrainDeSaga model to train.
            dataset: The StoryDataset to train on.
            config: Training configuration containing hyperparameters.
        """
        self.model = model
        self.dataset = dataset
        self.config = config

        # Set up optimizer.
        # AdamW is a variant of Adam that includes weight decay regularization,
        # which helps prevent overfitting by penalizing large weights.
        self.optimizer = optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Set up loss function.
        # Cross-entropy loss is standard for language modeling tasks as it measures
        # the difference between the predicted token distribution and the actual next token.
        self.loss_function = nn.losses.cross_entropy

        # Training state variables.
        # These track progress and store the best model during training.
        self.global_step = 0
        self.best_loss = float('inf')

        # Create checkpoint directory if it doesn't exist.
        os.makedirs(config.checkpoint_directory, exist_ok=True)

    def compute_loss(
        self,
        input_ids: mx.array,
        target_ids: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """
        Compute the loss for a batch of examples.

        This function will be called by nn.value_and_grad after the model
        has already been updated with the parameters to use.

        Args:
            input_ids: Input token IDs of shape [batch_size, sequence_length].
            target_ids: Target token IDs of shape [batch_size, sequence_length].

        Returns:
            Tuple of (loss, logits), where loss is a scalar and logits have shape
            [batch_size, sequence_length, vocabulary_size].
        """
        # Forward pass through the model.
        # The model has already been updated with the parameters by nn.value_and_grad
        logits = self.model(input_ids)

        # Reshape for cross entropy loss.
        # The cross_entropy function expects logits of shape [batch_size * sequence_length, vocabulary_size]
        # and targets of shape [batch_size * sequence_length].
        batch_size, sequence_length, vocabulary_size = logits.shape
        logits_reshaped = logits.reshape(-1, vocabulary_size)
        targets_reshaped = target_ids.reshape(-1)

        # Compute cross-entropy loss.
        # This measures how well the model predicts the next token at each position.
        loss = self.loss_function(logits_reshaped, targets_reshaped, reduction='mean')

        # Return loss and original-shaped logits.
        return loss, logits

    def train_step(
        self,
        input_ids: mx.array,
        target_ids: mx.array
    ) -> Dict[str, float]:
        """
        Perform a single training step.

        This method computes the loss and gradients for a batch of examples,
        updates the model parameters using the optimizer, and returns metrics.

        Args:
            input_ids: Input token IDs of shape [batch_size, sequence_length].
            target_ids: Target token IDs of shape [batch_size, sequence_length].

        Returns:
            Dictionary with training metrics, including the loss value.
        """
        # Define loss and gradient function.
        # This function computes both the loss value and the gradients with respect to model parameters.
        loss_and_gradient_function = nn.value_and_grad(self.model, self.compute_loss)

        # Compute loss and gradients.
        # Note: We don't pass parameters explicitly here because nn.value_and_grad handles that internally
        (loss, _), gradients = loss_and_gradient_function(input_ids, target_ids)

        # Update model parameters using the optimizer.
        # The optimizer applies the computed gradients to the model parameters,
        # adjusting them to minimize the loss function.
        self.optimizer.update(self.model, gradients)

        # Return metrics dictionary.
        return {"loss": loss.item()}

    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the entire dataset.

        This method evaluates the model's performance on the dataset without
        updating the model parameters, providing an unbiased assessment of
        the model's current capabilities.

        Returns:
            Dictionary with validation metrics, including the average loss.
        """
        total_loss = 0.0
        num_batches = 0

        # Get sequential batches for evaluation.
        for input_ids, target_ids in self.dataset.get_sequential_batches(self.config.batch_size):
            # Compute loss without updating parameters.
            # We use the compute_loss function directly since we don't need gradients for validation
            loss, _ = self.compute_loss(input_ids, target_ids)
            total_loss += loss.item()
            num_batches += 1

        # Calculate average loss across all batches.
        average_loss = total_loss / max(1, num_batches)

        return {"validation_loss": average_loss}

    def save_checkpoint(self, path: str) -> None:
        """
        Save a model checkpoint.

        This method saves the model parameters, optimizer state, and training progress
        to a file, allowing training to be resumed later or the model to be used for inference.

        Args:
            path: Path to save the checkpoint.
        """
        # Flatten the model parameters.
        model_parameters = dict(tree_flatten(self.model.parameters()))

        # Flatten optimizer state.
        optimizer_state = dict(tree_flatten(self.optimizer.state))

        # Prefix optimizer state keys to avoid conflicts.
        optimizer_state = {f"optimizer_{k}": v for k, v in optimizer_state.items()}

        # Create a dictionary with all the state we need to save.
        checkpoint = {
            **model_parameters,
            **optimizer_state,
            "global_step": mx.array(self.global_step),
            "best_loss": mx.array(self.best_loss)
        }

        # Save the checkpoint to a file.
        mx.savez(path, **checkpoint)

    def load_checkpoint(self, path: str) -> None:
        """
        Load a model checkpoint.

        This method loads a previously saved checkpoint, restoring the model parameters,
        optimizer state, and training progress to continue training or perform inference.

        Args:
            path: Path to load the checkpoint from.
        """
        # Load the checkpoint from the file.
        checkpoint = mx.load(path)

        # Extract model parameters (keys without 'optimizer_' prefix).
        model_parameters = {k: v for k, v in checkpoint.items()
            if not k.startswith("optimizer_") and k not in ["global_step", "best_loss"]}

        # Extract optimizer state (keys with 'optimizer_' prefix).
        optimizer_state = {k[4:]: v for k, v in checkpoint.items() if k.startswith("optimizer_")}

        # Restore model parameters.
        self.model.update(tree_unflatten(list(model_parameters.items())))

        # Restore optimizer state.
        self.optimizer.state = tree_unflatten(list(optimizer_state.items()))

        # Restore training progress.
        self.global_step = checkpoint["global_step"].item()
        self.best_loss = checkpoint["best_loss"].item()

    def train(self, callback: Optional[Callable[[Dict[str, float]], None]] = None) -> None:
        """
        Train the model for the specified number of epochs.

        This method implements the main training loop, iterating through the dataset
        for multiple epochs, updating the model parameters, and periodically evaluating
        and saving the model.

        Args:
            callback: Optional callback function called after each epoch with metrics.
                     This can be used for logging, visualization, or early stopping.
        """
        print(f"Starting training for {self.config.max_epochs} epochs.")

        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0

            # Train for one epoch.
            for input_ids, target_ids in self.dataset.get_sequential_batches(self.config.batch_size):
                # Perform training step.
                metrics = self.train_step(input_ids, target_ids)
                epoch_loss += metrics["loss"]
                num_batches += 1
                self.global_step += 1

                # Log progress at specified intervals.
                if self.global_step % self.config.log_interval == 0:
                    print(f"Step {self.global_step}, Loss: {metrics['loss']:.4f}")

            # Compute average loss for the epoch.
            average_epoch_loss = epoch_loss / max(1, num_batches)
            epoch_time = time.time() - start_time

            # Validate the model.
            validation_metrics = self.validate()
            validation_loss = validation_metrics["validation_loss"]

            # Log epoch results.
            print(f"Epoch {epoch+1}/{self.config.max_epochs}, "
                  f"Loss: {average_epoch_loss:.4f}, "
                  f"Validation Loss: {validation_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")

            # Save checkpoint if this is the best model so far.
            if validation_loss < self.best_loss:
                self.best_loss = validation_loss
                best_model_path = os.path.join(self.config.checkpoint_directory, "best_model.npz")
                self.save_checkpoint(best_model_path)
                print(f"Saved best model with validation loss: {validation_loss:.4f}")

            # Save regular checkpoint for this epoch.
            checkpoint_path = os.path.join(self.config.checkpoint_directory, f"epoch_{epoch+1}.npz")
            self.save_checkpoint(checkpoint_path)

            # Call callback if provided.
            if callback is not None:
                callback({
                    "epoch": epoch + 1,
                    "loss": average_epoch_loss,
                    "validation_loss": validation_loss,
                    "time": epoch_time
                })
