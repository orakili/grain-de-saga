"""
BPE Tokenizer implementation for the Grain de Saga model.

This module provides a Byte-Pair Encoding (BPE) tokenizer for processing text data.
BPE is a subword tokenization algorithm that iteratively merges the most frequent
pairs of bytes or characters to form a vocabulary of subword units.
"""

import regex
import os
import json
import collections
from typing import List, Dict, Tuple, Set, Optional, Union, Iterator


class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) tokenizer.

    BPE tokenization works by starting with a vocabulary of individual characters or bytes,
    then iteratively merging the most frequent adjacent pairs until the desired vocabulary
    size is reached. This approach effectively balances vocabulary size with the ability
    to represent any text, handling rare words by breaking them into subword units.
    """

    def __init__(self, vocabulary_size: int = 8000):
        """
        Initialize the BPE tokenizer.

        Args:
            vocabulary_size: Maximum vocabulary size, including special tokens and base characters.
        """
        self.vocabulary_size = vocabulary_size

        # Special tokens that have special meaning in the model.
        self.pad_token = "[PAD]"  # Used for padding sequences to the same length.
        self.unk_token = "[UNK]"  # Used for unknown tokens not in vocabulary.
        self.bos_token = "[BOS]"  # Beginning of sequence marker.
        self.eos_token = "[EOS]"  # End of sequence marker.

        # Dictionary mapping tokens to their integer IDs.
        self.token_to_id: Dict[str, int] = {}

        # Dictionary mapping integer IDs back to tokens.
        self.id_to_token: Dict[int, str] = {}

        # Dictionary of BPE merges, mapping pairs of tokens to their merged form.
        self.merges: Dict[Tuple[str, str], str] = {}

        # Pattern for basic word splitting, keeping punctuation as separate tokens.
        # This pre-tokenization step helps establish word boundaries before BPE.
        self.pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # Initialize with special tokens.
        self._add_special_tokens()

        # Initialize with basic byte vocabulary (0-255).
        # This ensures we can encode any possible character.
        self._initialize_byte_vocabulary()

    def _add_special_tokens(self) -> None:
        """
        Add special tokens to the vocabulary.

        Special tokens have reserved IDs at the beginning of the vocabulary.
        """
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def _initialize_byte_vocabulary(self) -> None:
        """
        Initialize the vocabulary with byte tokens (0-255).

        Each byte is represented as a string of its corresponding character or
        a special representation for control characters.
        """
        # Start after special tokens.
        next_id = len(self.token_to_id)

        # Add single-byte tokens (0-255).
        for b in range(256):
            # Convert byte to a string representation.
            # Printable ASCII characters are represented as-is.
            # Non-printable characters are represented as hex codes.
            if 32 <= b <= 126:
                token = bytes([b]).decode('ascii')
            else:
                token = f"<{b:02X}>"  # Hex representation for control characters.

            # Add to vocabulary if not already present.
            if token not in self.token_to_id:
                self.token_to_id[token] = next_id
                self.id_to_token[next_id] = token
                next_id += 1

    def _get_pairs(self, word: List[str]) -> collections.Counter:
        """
        Count frequencies of adjacent pairs in a word.

        Args:
            word: List of tokens representing a word.

        Returns:
            Counter of adjacent token pairs.
        """
        pairs = collections.Counter()
        prev_token = word[0]

        for token in word[1:]:
            pairs[(prev_token, token)] += 1
            prev_token = token

        return pairs

    def _merge_pair(self, pair: Tuple[str, str], token_list: List[str]) -> List[str]:
        """
        Merge all occurrences of a token pair in a list.

        Args:
            pair: Tuple of two tokens to merge.
            token_list: List of tokens.

        Returns:
            Updated list with the pair merged.
        """
        first, second = pair
        merged_token = first + second
        i = 0

        while i < len(token_list) - 1:
            if token_list[i] == first and token_list[i + 1] == second:
                token_list[i] = merged_token
                del token_list[i + 1]
            else:
                i += 1

        return token_list

    def train(self, texts: List[str], min_frequency: int = 2, verbose: bool = False) -> None:
        """
        Train the BPE tokenizer on a corpus of texts.

        Args:
            texts: List of text samples to train on.
            min_frequency: Minimum frequency for a pair to be considered for merging.
            verbose: Whether to print progress information.
        """
        if verbose:
            print("Pre-tokenizing texts...")

        # Pre-tokenize texts into words.
        words: Dict[str, int] = collections.Counter()
        for text in texts:
            # Find all matches of our pattern in the text.
            for match in regex.finditer(self.pattern, text):
                word = match.group(0)
                if word:
                    words[word] += 1

        # Convert words to list of characters/bytes.
        word_tokens: Dict[str, List[str]] = {}
        for word, count in words.items():
            # Convert each character to its byte representation.
            tokens = []
            for char in word:
                byte_val = char.encode('utf-8')[0]  # Take first byte for simplicity.
                if 32 <= byte_val <= 126:
                    tokens.append(char)
                else:
                    tokens.append(f"<{byte_val:02X}>")
            word_tokens[word] = tokens

        # Calculate initial vocabulary size after special tokens and bytes.
        initial_vocab_size = len(self.token_to_id)
        merges_to_learn = self.vocabulary_size - initial_vocab_size

        if verbose:
            print(f"Learning {merges_to_learn} BPE merges...")

        # Iteratively merge most frequent pairs.
        for i in range(merges_to_learn):
            # Count all pairs across all words.
            pair_counts = collections.Counter()
            for word, tokens in word_tokens.items():
                word_count = words[word]
                pairs = self._get_pairs(tokens)
                for pair, count in pairs.items():
                    pair_counts[pair] += count * word_count

            # If no more pairs to merge, we're done.
            if not pair_counts:
                break

            # Find the most frequent pair.
            best_pair = max(pair_counts.items(), key=lambda x: x[1])
            pair, count = best_pair

            # Skip pairs that occur less than min_frequency.
            if count < min_frequency:
                break

            # Create new token from the pair.
            first, second = pair
            new_token = first + second

            # Add the merged token to the vocabulary.
            next_id = len(self.token_to_id)
            self.token_to_id[new_token] = next_id
            self.id_to_token[next_id] = new_token

            # Record the merge operation.
            self.merges[pair] = new_token

            # Apply the merge to all words.
            for word in word_tokens:
                word_tokens[word] = self._merge_pair(pair, word_tokens[word])

            if verbose and (i + 1) % 100 == 0:
                print(f"Learned {i + 1}/{merges_to_learn} merges. Current vocab size: {len(self.token_to_id)}")

        if verbose:
            print(f"Final vocabulary size: {len(self.token_to_id)}")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        Apply BPE encoding to a single word.

        Args:
            word: Word to tokenize.

        Returns:
            List of BPE tokens.
        """
        # Convert word to list of characters/bytes.
        tokens = []
        for char in word:
            byte_val = char.encode('utf-8')[0]  # Take first byte for simplicity.
            if 32 <= byte_val <= 126:
                tokens.append(char)
            else:
                tokens.append(f"<{byte_val:02X}>")

        # Apply merges iteratively.
        while len(tokens) > 1:
            pairs = self._get_pairs(tokens)
            if not pairs:
                break

            # Find the highest-priority merge.
            next_pair = None
            for pair in pairs:
                if pair in self.merges:
                    next_pair = pair
                    break

            # If no applicable merge found, we're done.
            if next_pair is None:
                break

            # Apply the merge.
            tokens = self._merge_pair(next_pair, tokens)

        return tokens

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into BPE tokens.

        Args:
            text: Text to tokenize.

        Returns:
            List of BPE tokens.
        """
        tokens = []

        # Apply pre-tokenization pattern.
        for match in regex.finditer(self.pattern, text):
            word = match.group(0)
            if not word:
                continue

            # Apply BPE to each word.
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text into token IDs.

        Args:
            text: Text to encode.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of token IDs.
        """
        # Tokenize the text.
        tokens = self.tokenize(text)

        # Convert tokens to IDs.
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Handle unknown tokens.
                ids.append(self.token_to_id[self.unk_token])

        # Add special tokens if requested.
        if add_special_tokens:
            ids = [self.token_to_id[self.bos_token]] + ids + [self.token_to_id[self.eos_token]]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text.
        """
        # Convert IDs to tokens.
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                token = self.id_to_token[id]

                # Skip special tokens if requested.
                if skip_special_tokens and token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                    continue

                tokens.append(token)

        # Merge tokens back into text.
        # This is a simplified approach that works for basic BPE.
        # A more sophisticated approach would handle merging subword units properly.
        text = ''.join(tokens)

        # Replace hex-encoded bytes with their character representation.
        def replace_hex(match):
            hex_val = match.group(1)
            byte_val = int(hex_val, 16)
            try:
                return bytes([byte_val]).decode('utf-8')
            except UnicodeDecodeError:
                return f"<{hex_val}>"

        text = regex.sub(r'<([0-9A-F]{2})>', replace_hex, text)

        return text

    def save(self, path: str) -> None:
        """
        Save the tokenizer vocabulary and merges to a file.

        Args:
            path: Path to save the tokenizer.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert merges to a serializable format.
        merges_list = []
        for (first, second), merged in self.merges.items():
            merges_list.append((first, second, merged))

        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocabulary_size': self.vocabulary_size,
                'token_to_id': self.token_to_id,
                'merges': merges_list,
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
        """
        Load a tokenizer from a file.

        Args:
            path: Path to load the tokenizer from.

        Returns:
            Loaded tokenizer.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create a new tokenizer instance.
        tokenizer = cls(vocabulary_size=data['vocabulary_size'])

        # Update special tokens if they differ from defaults.
        tokenizer.pad_token = data['pad_token']
        tokenizer.unk_token = data['unk_token']
        tokenizer.bos_token = data['bos_token']
        tokenizer.eos_token = data['eos_token']

        # Load vocabulary.
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(v): k for k, v in data['token_to_id'].items()}

        # Load merges.
        tokenizer.merges = {}
        for first, second, merged in data['merges']:
            tokenizer.merges[(first, second)] = merged

        return tokenizer
