# Grain de Saga

A tiny language model for generating children's stories, built with MLX for Apple Silicon.

## Overview

Grain de Saga is a lightweight transformer-based language model designed specifically for generating short stories for children. With approximately 3 million parameters, it's optimized to run efficiently on Apple Silicon hardware while still producing creative and engaging content.

## Features

- Transformer-based architecture with self-attention
- Optimized for Apple Silicon using MLX
- Generates children's stories based on themes, tones, and characters
- Trainable in under an hour on a MacBook Air M2

## Requirements

- Python 3.13+
- MLX 0.24.1+
- macOS with Apple Silicon (M1/M2/M3)

## Installation

Clone the repository and install the package:

```sh
git clone https://github.com/yourusername/grain_de_saga.git
cd grain_de_saga
pip install -e .
```

## Usage

### Training

To train the model on your own data:

```sh
python scripts/train.py --data_dir path/to/stories --output_dir output
```

### Generating Stories

To generate stories with a trained model:

```sh
python scripts/generate.py --model_path output/final_model.npz --tokenizer_path output/tokenizer.json --theme adventure --tone cheerful --genre fantasy --characters "Alice,Bob"
```

## Model Architecture

Grain de Saga uses a standard transformer architecture with:

- Vocabulary size: 8,000 tokens
- Context length: 512 tokens
- Embedding dimension: 128
- Number of layers: 3
- Number of attention heads: 4
- Feed-forward dimension: 384
- Total parameters: ~2.5 million

## License

0BSD License

## Acknowledgements

This project was inspired by and builds upon several key resources:

- **[TinyStories Dataset V2 cleaned](https://huggingface.co/datasets/fhswf/TinyStoriesV2_cleaned)**: Our model is designed to be trained on the TinyStories V2 dataset, a synthetic collection of simple children's stories generated by GPT-4. This dataset was specifically created to contain vocabulary understandable by 3-4 year olds while preserving essential language elements like grammar, reasoning, and facts. As demonstrated in research, models trained on TinyStories can generate coherent narratives even with limited parameters. This dataset is derived from the original [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories/).

- **[Simple Stories Model](https://huggingface.co/broskicodes/simple-stories-4M)**: We drew inspiration from the Simple Stories series of models by broskicodes, particularly the 4M parameter implementation that demonstrated the feasibility of training small language models for children's story generation. Their work showed that even with just 4 decoder layers and 2 attention heads, a model trained on a relatively small corpus (~50MB) could produce semi-coherent stories.

- **[LightLM](https://github.com/dongyuanjushi/LightLM)**: Our implementation methodology was influenced by the LightLM project, which provided insights into building lightweight language models from scratch with a deep but narrow architecture.

- **[MLX Framework](https://github.com/ml-explore/mlx)**: We utilized Apple's MLX framework for efficient model training and inference on Apple Silicon hardware.

The approach of building small, focused language models aligns with research showing that coherent text generation doesn't necessarily require massive models when the training data is carefully curated for a specific domain.

