import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class ModelArgs:
    vocab_size: int
    context_length: int
    embedding_dim: int
    n_heads: int
    n_layers: int
    hidden_dim: int
    n_kv_groups: int
    rope_theta: float
    dtype: jnp.dtype = jnp.bfloat16
    rope_freq: dict
