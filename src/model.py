"""
Model architecture for the custom made Large Language Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

class SwiGLU(nn.Module):
    """
    SwiGLU activation function with gating mechanism (currently not used, SiLU is used instead)
    """
    def __init__(self, config):
        super().__init__()
        self.w = nn.Linear(config.embed_dim, 8 * config.embed_dim)

    def forward(self, x):
        out = self.w(x)
        swish, gate = out.chunk(2, dim=-1)
        return F.silu(swish) * gate
    
class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head latent attention (from Deepseek V2 paper)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.embed_dim
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        # Low-rank KV compression
        self.W_dkv = nn.Linear(self.hidden_size, config.kv_lora_rank)
        self.W_uk = nn.Linear(config.kv_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.W_uv = nn.Linear(config.kv_lora_rank, self.num_heads * self.head_dim, bias=False)

        # Query compression
        self.W_dq = nn.Linear(self.hidden_size, config.q_lora_rank)
        self.W_uq = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)

        # RoPE projections (applied after low-rank compression)
        self.rope_q = RotaryEmbedding(self.head_dim)  # RoPE for queries
        self.rope_k = RotaryEmbedding(self.head_dim)  # RoPE for keys

        # Output projection
        self.W_o = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)

        # RMSNorm for KV compression
        self.norm_c = nn.RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)

    def forward(self, h, cache=None):
        batch_size, seq_len, _ = h.shape

        # Low-rank KV compression
        c_kv = self.W_dkv(h)
        c_kv = self.norm_c(c_kv)
        k = self.W_uk(c_kv).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.W_uv(c_kv).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Query compression
        c_q = self.W_dq(h)
        q = self.W_uq(c_q).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE directly to compressed queries and keys
        q = self.rope_q.rotate_queries_or_keys(q)
        k = self.rope_k.rotate_queries_or_keys(k)

        # Compute attention (no concatenation needed)
        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.config.attention_dropout,
            is_causal=True,
        )

        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.W_o(attn_output)

class FeedForward(nn.Module):
    """
    Feed-forward network
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.embed_dim, 4 * config.embed_dim)
        self.fc2 = nn.Linear(4 * config.embed_dim, config.embed_dim)
        self.act = nn.SiLU()
        self.norm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.norm(x))))
    
class TransformerBlock(nn.Module):
    """
    Transformer block
    """
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadLatentAttention(config)
        self.ff = FeedForward(config)
        self.norm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.ff(x)
        return x

class Model(nn.Module):
    """
    Custom made Large Language Model
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text from the model
        """
        for _ in range(max_new_tokens):
            idx_input = idx if idx.size(1) <= self.config.max_position_embeddings else idx[:, -self.config.max_position_embeddings:]
            logits = self(idx_input)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

