"""
Configuration for the model and training.
"""
import torch
from dataclasses import dataclass

@dataclass
class Config:
    # Device
    device: str = "mps"
    dtype: torch.dtype = torch.bfloat16

    # Model
    vocab_size: int = 65536
    attention_bias: bool = False
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    max_position_embeddings: int = 1024
    embed_dim: int = 768
    num_layers: int = 6
    num_attention_heads: int = 8
    kv_lora_rank: int = 32
    q_lora_rank: int = 64
    qk_rope_head_dim: int = 24
    rms_norm_eps: float = 1e-6

    # Tokenizer
    tokenizer_name: str = "PleIAs/Pleias-350m-Preview"

    # Training
    dataset_name: str = "PleIAs/common_corpus"

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    batch_size: int = 2
    
    # Training schedule
    max_steps: int = 3000
    num_epochs: int = 10
    log_every: int = 100
    
    # Checkpointing
    ckpt_dir: str = "checkpoints"
