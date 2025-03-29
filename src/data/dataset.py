"""
Dataset handling for the Grain de Saga model.

This module provides dataset loading and processing functionality,
specifically designed for handling children's stories and preparing
them for training the Grain de Saga language model.
"""

from typing import List, Dict, Tuple, Optional, Iterator
import os
import random
import mlx.core as mx

from .tokenizer import BPETokenizer


class StoryDataset:
    """
    Dataset for children's stories.

    This class handles the loading, tokenization, and batching of story data
    for training the Grain de Saga language model. It supports sliding window
    tokenization to create multiple training examples from each story.
    """

    def __init__(
        self,
        tokenizer: BPETokenizer,
        max_sequence_length: int = 512,
        stride: int = 128
    ):
        """
        Initialize the dataset.

        Args:
            tokenizer: BPETokenizer to use for encoding texts.
            max_sequence_length: Maximum sequence length for each example.
            stride: Stride for sliding window tokenization.
        """
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.stride = stride
        self.examples: List[List[int]] = []

    def load_texts(self, texts: List[str]):
        """
        Load and tokenize texts.

        This method processes each text in the input list, tokenizes it,
        and creates multiple training examples using a sliding window approach.

        Args:
            texts: List of text samples to load.
        """
        for text in texts:
            # Encode the text using the BPE tokenizer.
            tokens = self.tokenizer.encode(text)

            # Create examples with sliding window.
            for i in range(0, len(tokens) - self.max_sequence_length + 1, self.stride):
                example = tokens[i:i + self.max_sequence_length]
                if len(example) == self.max_sequence_length:
                    self.examples.append(example)

            # Add the last chunk if it's not already included.
            if len(tokens) > self.stride and len(tokens) % self.stride != 0:
                last_chunk = tokens[-self.max_sequence_length:]
                if len(last_chunk) == self.max_sequence_length:
                    self.examples.append(last_chunk)

    def load_from_directory(self, directory: str, file_extension: str = ".txt"):
        """
        Load texts from a directory.

        This method reads all files with the specified extension in the given directory
        and loads them as separate stories into the dataset.

        Args:
            directory: Directory containing text files.
            file_extension: File extension to look for (default is ".txt").
        """
        texts = []
        for filename in os.listdir(directory):
            if filename.endswith(file_extension):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    texts.append(file.read())

        self.load_texts(texts)

    def __len__(self) -> int:
        """
        Get the number of examples in the dataset.

        Returns:
            The total number of training examples.
        """
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array]:
        """
        Get an example from the dataset.

        This method returns a single training example, split into input and target.
        The input is all tokens except the last one, and the target is all tokens
        except the first one, allowing for next-token prediction training.

        Args:
            idx: Index of the example.

        Returns:
            Tuple of (input_ids, target_ids), both as MLX arrays.
        """
        tokens = self.examples[idx]

        # Input is all tokens except the last one.
        input_ids = mx.array(tokens[:-1])

        # Target is all tokens except the first one.
        target_ids = mx.array(tokens[1:])

        return input_ids, target_ids

    def get_batch(self, batch_size: int) -> Tuple[mx.array, mx.array]:
        """
        Get a random batch of examples.

        This method randomly samples a batch of examples from the dataset,
        which is useful for stochastic gradient descent training.

        Args:
            batch_size: Size of the batch.

        Returns:
            Tuple of (input_ids, target_ids), both as batched MLX arrays.
        """
        indices = random.sample(range(len(self)), min(batch_size, len(self)))

        # Get examples.
        examples = [self[i] for i in indices]

        # Stack into batches.
        inputs = mx.stack([ex[0] for ex in examples])
        targets = mx.stack([ex[1] for ex in examples])

        return inputs, targets

    def get_sequential_batches(self, batch_size: int) -> Iterator[Tuple[mx.array, mx.array]]:
        """
        Get sequential batches for the entire dataset.

        This method provides an iterator over the entire dataset in batches,
        which is useful for epoch-based training or evaluation.

        Args:
            batch_size: Size of each batch.

        Yields:
            Tuples of (input_ids, target_ids), both as batched MLX arrays.
        """
        indices = list(range(len(self)))
        random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            examples = [self[idx] for idx in batch_indices]

            # Stack into batches.
            inputs = mx.stack([ex[0] for ex in examples])
            targets = mx.stack([ex[1] for ex in examples])

            yield inputs, targets
