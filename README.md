# Custom Large Language Model (LLM) Implementation

A minimal implementation of a transformer-based Large Language Model (LLM) inspired by modern architectures like [Deepseek V2](https://arxiv.org/abs/2405.04434), [minGPT](https://github.com/karpathy/minGPT), and [nanoGPT](https://github.com/karpathy/nanoGPT). This project includes features like low-rank attention compression, SwiGLU activation, and rotary positional embeddings.

## Features
- **Multi-head latent attention** with low-rank compression for keys, values, and queries.
- **SwiGLU activation** for improved gating mechanisms. (SiLU currently used instead)
- **Rotary Positional Embeddings (RoPE)** for better positional encoding.
- **Lightweight and modular** design for easy experimentation.

## Project Files
Here's an overview of the key files and directories in this project:
- `src/model.py`: Core implementation of the model, including attention mechanisms, feed-forward layers, and the transformer architecture.
- `src/trainer.py`: Training loop for the model.
- `src/main.py`: Entry point for running the model (inference).
- `src/config.py`: Configuration utilities for model hyperparameters and training config.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/clement-cvll/open-large-language-model
   cd open-large-language-model
   ```
2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage
### Training
To train the model, use the provided script:
```bash
python src/trainer.py
```

### Inference
Generate text with the trained model:
```bash
python src/main.py
```

### Configuration
Modify `src/config.py` to adjust hyperparameters like:
- `embed_dim`: Embedding dimension.
- `num_attention_heads`: Number of attention heads.
- `num_layers`: Number of transformer layers.
- `device`: Torch device (`mps` is default).

## Tokenizer and Dataset
### Tokenizer
The tokenizer used in this project is the **Pleias-350m-Preview** tokenizer from Hugging Face ([link](https://huggingface.co/PleIAs/Pleias-350m-Preview)).

### Dataset
The model is designed to work with the **Common Corpus** dataset ([link](https://huggingface.co/datasets/PleIAs/common_corpus)), a large, open, and permissively licensed multilingual dataset.

## Inspiration
This project draws inspiration from:
- **minGPT**: A minimal PyTorch re-implementation of GPT by Andrej Karpathy.
- **nanoGPT**: A simplified and efficient GPT implementation, also by Andrej Karpathy.
- **Deepseek V2**: For modern architectural choices like low-rank attention and SwiGLU.

## License
[MIT](https://choosealicense.com/licenses/mit/)