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
    parser.add_argument("--data_directory", type=str, required=True,
                        help="Directory containing training data files")
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


def load_training_data(data_directory: str) -> List[str]:
    """
    Load training data from text files in a directory.

    Args:
        data_directory: Directory containing text files.

    Returns:
        List of text samples loaded from the files.
    """
    text_samples = []

    # Iterate through all files in the directory
    for filename in os.listdir(data_directory):
        # Only process text files
        if filename.endswith('.txt'):
            file_path = os.path.join(data_directory, filename)

            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                text_samples.append(file.read())

    return text_samples


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

    # Load training data
    print(f"Loading data from {args.data_directory}")
    text_samples = load_training_data(args.data_directory)

    # Check if we have any data
    if not text_samples:
        print("No training data found. Exiting.")
        return

    print(f"Loaded {len(text_samples)} text samples")

    # Initialize tokenizer
    print("Initializing tokenizer")
    tokenizer = BPETokenizer(vocabulary_size=args.vocabulary_size)

    # Train tokenizer on data
    print("Training tokenizer")
    tokenizer.train(text_samples, verbose=True)

    # Save tokenizer
    tokenizer_path = os.path.join(args.output_directory, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")

    # Initialize dataset
    print("Preparing dataset")
    dataset = StoryDataset(
        tokenizer=tokenizer,
        max_sequence_length=args.context_length
    )
    dataset.load_texts(text_samples)
    print(f"Created dataset with {len(dataset)} examples")

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

    # Print model summary
    total_parameters = sum(parameter.size for parameter in model.parameters().values())
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
