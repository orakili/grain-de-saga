# Grain de Saga Documentation

This directory contains additional documentation for the Grain de Saga language model.

## Contents

- [Model Architecture](model_architecture.md)
- [Training Guide](training_guide.md)
- [Generation Guide](generation_guide.md)
- [API Reference](api_reference.md)

## Model Architecture

Grain de Saga is a transformer-based language model with approximately 2.5 million parameters. The architecture consists of:

1. **Embedding Layer**: Converts token IDs to vectors and adds positional information
2. **Transformer Blocks**: 3 layers of transformer blocks with self-attention and feed-forward networks
3. **Output Layer**: Projects hidden states to vocabulary logits

For more details, see [Model Architecture](model_architecture.md).

## Training

The model is designed to be trained on a dataset of children's stories in under an hour on a MacBook Air M2. The training process involves:

1. Tokenizing the input texts
2. Creating a dataset with sliding window examples
3. Training the model with cross-entropy loss

For more details, see [Training Guide](training_guide.md).

## Generation

Grain de Saga can generate children's stories based on specified themes, tones, genres, and characters. The generation process uses:

- Temperature sampling
- Top-k filtering
- Prompt-based conditioning

For more details, see [Generation Guide](generation_guide.md).
