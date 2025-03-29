"""
Generation script for the Grain de Saga model.

This script handles text generation using a trained model, allowing users
to generate children's stories based on prompts, themes, and other parameters.
"""

import argparse
import os
import sys
from typing import List, Optional

# Add the project root to the Python path
script_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_directory, '..'))
sys.path.insert(0, project_root)

import mlx.core as mx

from src.model.grain_de_saga import GrainDeSaga
from src.data.tokenizer import BPETokenizer
from src.utils.config import ModelConfig


def parse_arguments():
    """
    Parse command line arguments for the generation script.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate text with the Grain de Saga model")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the saved tokenizer")

    # Generation arguments
    parser.add_argument("--prompt", type=str, default="",
                        help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum length of generated text (including prompt)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher = more random, lower = more deterministic)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter (restricts sampling to k most likely tokens)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of different samples to generate")

    # Story parameters
    parser.add_argument("--theme", type=str, default="adventure",
                        help="Theme for the story (e.g., adventure, friendship, nature)")
    parser.add_argument("--tone", type=str, default="cheerful",
                        help="Tone of the story (e.g., cheerful, mysterious, educational)")
    parser.add_argument("--genre", type=str, default="fantasy",
                        help="Genre of the story (e.g., fantasy, science fiction, fable)")
    parser.add_argument("--characters", type=str, default="",
                        help="Main characters for the story (comma-separated)")

    return parser.parse_args()


def load_model(model_path: str, config: ModelConfig) -> GrainDeSaga:
    """
    Load a trained model from a checkpoint.

    Args:
        model_path: Path to the model checkpoint file.
        config: Model configuration.

    Returns:
        Loaded GrainDeSaga model.
    """
    # Initialize a new model with the provided configuration
    model = GrainDeSaga(config)

    # Load the model weights and state from the checkpoint
    checkpoint = mx.load(model_path)

    # Update the model with the loaded parameters
    model.update(checkpoint["model"])

    return model


def create_story_prompt(
    theme: str,
    tone: str,
    genre: str,
    characters: Optional[str] = None
) -> str:
    """
    Create a formatted prompt for story generation based on user parameters.

    This function constructs a natural language prompt that instructs the model
    to generate a story with the specified characteristics.

    Args:
        theme: Theme of the story (e.g., adventure, friendship).
        tone: Tone of the story (e.g., cheerful, mysterious).
        genre: Genre of the story (e.g., fantasy, science fiction).
        characters: Optional comma-separated list of character names.

    Returns:
        Formatted prompt string.
    """
    # Start with the basic prompt structure
    prompt = f"Write a {tone} {genre} story about {theme}"

    # Add characters if provided
    if characters:
        character_list = characters.split(",")
        if len(character_list) == 1:
            # Single character
            prompt += f" featuring {character_list.strip()}"
        else:
            # Multiple characters - format as "X, Y, and Z"
            formatted_characters = ", ".join(c.strip() for c in character_list[:-1])
            prompt += f" featuring {formatted_characters} and {character_list[-1].strip()}"

    # Add a colon and newlines to separate the instruction from the generated content
    prompt += ":\n\n"

    return prompt


def main():
    """
    Main generation function.

    This function handles the complete generation process:
    1. Parse command line arguments
    2. Load the tokenizer and model
    3. Create or use a prompt
    4. Generate text samples
    5. Display the results
    """
    # Parse command line arguments
    args = parse_arguments()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = BPETokenizer.load(args.tokenizer_path)

    # Create model configuration
    # We only need to specify the vocabulary size and context length,
    # as the other parameters will be loaded from the checkpoint
    model_config = ModelConfig(
        vocabulary_size=len(tokenizer.token_to_id),
        context_length=args.max_length
    )

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, model_config)

    # Create or use provided prompt
    if args.prompt:
        # Use the prompt directly if provided
        prompt = args.prompt
    else:
        # Otherwise, create a prompt based on the story parameters
        prompt = create_story_prompt(
            theme=args.theme,
            tone=args.tone,
            genre=args.genre,
            characters=args.characters
        )

    # Display the prompt that will be used
    print("\nGenerating with prompt:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)

    # Generate multiple samples if requested
    for sample_index in range(args.num_samples):
        print(f"\nSample {sample_index+1}:")

        # Encode the prompt to token IDs
        # We need to wrap it in a batch dimension (hence the [])
        input_ids = mx.array([tokenizer.encode(prompt)])

        # Generate text
        # The model will continue from the prompt up to max_length tokens
        generated_ids = model.generate(
            input_ids,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )

        # Decode the generated token IDs back to text
        generated_text = tokenizer.decode(generated_ids.tolist())

        # Display the generated text
        print(generated_text)
        print("-" * 40)


if __name__ == '__main__':
    main()
