"""
Training script for the Grain de Saga model.

This script handles the complete training process for the model, including
data loading, tokenization, model initialization, and training loop execution.
It provides command-line arguments for configuring various aspects of the training.
"""

import argparse
import os
import sys
import time
from typing import List

# Add the project root to the Python path
script_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_directory, '..'))
sys.path.insert(0, project_root)

import mlx.core as mx
from mlx.utils import tree_flatten

from src.model.grain_de_saga import GrainDeSaga
from src.data.dataset import StoryDataset
from src.data.tokenizer import BPETokenizer
from src.training.trainer import Trainer
from src.utils.config import ModelConfig, TrainingConfig


def parse_arguments():
    """
    Parse command line arguments for the training script.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Train the Grain de Saga model")

    # Data arguments
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data_directory", type=str,
                            help="Directory containing training data files")
    data_group.add_argument("--huggingface_dataset", type=str,
                            help="Name of the Hugging Face dataset to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use from the dataset")
    parser.add_argument("--output_directory", type=str, default="output",
                        help="Directory to save model and tokenizer")

    # Model arguments
    parser.add_argument("--vocabulary_size", type=int, default=8000,
                        help="Size of the vocabulary")
    parser.add_argument("--context_length", type=int, default=512,
                        help="Maximum context length for the model")
    parser.add_argument("--embedding_dimension", type=int, default=128,
                        help="Dimension of the token embeddings")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads per layer")
    parser.add_argument("--feed_forward_dimension", type=int, default=384,
                        help="Dimension of the feed-forward network")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                        help="Dropout rate for regularization")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate for optimization")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Maximum number of training epochs")
    parser.add_argument("--checkpoint_directory", type=str, default="checkpoints",
                        help="Directory for saving model checkpoints")

    return parser.parse_args()


def main():
    """
    Main training function.

    This function orchestrates the complete training process:
    1. Parse command line arguments
    2. Load and prepare the training data
    3. Initialize and train the tokenizer
    4. Create the dataset
    5. Initialize the model
    6. Train the model
    7. Save the final model
    """
    # Parse command line arguments
    args = parse_arguments()

    # Create output directories
    os.makedirs(args.output_directory, exist_ok=True)
    os.makedirs(args.checkpoint_directory, exist_ok=True)

    # Initialize tokenizer
    print("Initializing tokenizer")
    tokenizer = BPETokenizer(vocabulary_size=args.vocabulary_size)

    # Initialize dataset
    print("Preparing dataset")
    dataset = StoryDataset(
        tokenizer=tokenizer,
        max_sequence_length=args.context_length
    )

    # Load data using the appropriate method
    if args.data_directory:
        print(f"Loading data from {args.data_directory}")
        dataset.load_from_directory(args.data_directory, max_samples=args.max_samples)
    else:
        print(f"Loading data from Hugging Face dataset: {args.huggingface_dataset}")
        dataset.load_from_huggingface(args.huggingface_dataset, max_samples=args.max_samples)

    # Check if we have any data
    if len(dataset) == 0:
        print("No training data found. Exiting.")
        return

    print(f"Created dataset with {len(dataset)} examples")

    # Train tokenizer on the original texts
    print("Training tokenizer")
    tokenizer.train(dataset.get_raw_texts(), verbose=True)

    # Save tokenizer
    tokenizer_path = os.path.join(args.output_directory, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Initialize model configuration
    model_config = ModelConfig(
        vocabulary_size=args.vocabulary_size,
        context_length=args.context_length,
        embedding_dimension=args.embedding_dimension,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        feed_forward_dimension=args.feed_forward_dimension,
        dropout_rate=args.dropout_rate
    )

    # Initialize training configuration
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        checkpoint_directory=args.checkpoint_directory
    )

    # Initialize model
    print("Initializing model")
    model = GrainDeSaga(model_config)

    # Print model summary using tree_flatten
    total_parameters = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model initialized with {total_parameters:,} parameters")

    # Initialize trainer
    trainer = Trainer(model, dataset, training_config)

    # Start training
    print("Starting training")
    start_time = time.time()
    trainer.train()

    # Training complete
    total_training_time = time.time() - start_time
    print(f"Training completed in {total_training_time:.2f} seconds")

    # Save final model
    final_model_path = os.path.join(args.output_directory, "final_model.npz")
    trainer.save_checkpoint(final_model_path)
    print(f"Saved final model to {final_model_path}")


if __name__ == '__main__':
    main()
