# Code adapted from:
    # https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
    # Licensed under the Apache License, Version 2.0
# Copyright 2023-2025 Sebastian Raschka



import jax
import jax.numpy as jnp
from flax import nnx
from jax import checkpoint
from args import ModelArgs


class FeedForward(nnx.Module):
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        super().__init__()
        self.fc1 = nnx.Linear(args.embedding_dim, args.hidden_dim, rngs=rngs, use_bias=False, dtype=args.dtype)
        self.fc2 = nnx.Linear(args.embedding_dim, args.hidden_dim, rngs=rngs, use_bias=False, dtype=args.dtype)
        self.fc3 = nnx.Linear(args.hidden_dim, args.embedding_dim, rngs=rngs, use_bias=False, dtype=args.dtype)

    def __call__(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nnx.silu(x_fc1) * x_fc2
        return self.fc3(x)

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute inverse frequencies
    inv_freq = 1.0 / (theta_base ** (jnp.arange(0, head_dim, 2) / head_dim))

    # Frequency adjustments (optional)
    if freq_config is not None:
      low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
      high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

      wavelen = 2 * jnp.pi / inv_freq

      inv_freq_llama = jnp.where(
          wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
      )

      smooth_factor = (
          (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"])
          / (freq_config["high_freq_factor"] - freq_config["low_freq_factor"])
      )

      smoothed_inv_freq = (
          (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
      )

      is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
      inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

      inv_freq = inv_freq_llama

    # Position indices
    positions = jnp.arange(context_length)

    # Compute rotary angles
    angles = positions[:, None] * inv_freq[None, :]  # (context_len, head_dim // 2)

    # Duplicate for interleaved dimensions
    angles = jnp.concatenate([angles, angles], axis=-1)  # (context_len, head_dim)

    # Precompute sin/cos
    cos = jnp.cos(angles)
    sin = jnp.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x: (batch, heads, seq_len, head_dim)
    batch, heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dim must be even"

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    # Broadcast cos/sin: (1, 1, seq_len, head_dim)
    cos = cos[:seq_len, :][None, None, :, :]
    sin = sin[:seq_len, :][None, None, :, :]

    # Rotary transformation
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return (x * cos) + (rotated * sin)


class GroupedQueryAttention(nnx.Module):
    def __init__(
        self, d_in, d_out, context_length, num_heads, num_kv_groups,
        rope_base=10_000, rope_config=None, dtype=jnp.float32, rngs=nnx.Rngs
    ):
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_out // num_heads
        self.group_size = num_heads // num_kv_groups

        self.W_query = nnx.Linear(d_in, d_out, use_bias=False, dtype=dtype, rngs=rngs)
        self.W_key = nnx.Linear(d_in, num_kv_groups * self.head_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.W_value = nnx.Linear(d_in, num_kv_groups * self.head_dim, use_bias=False, dtype=dtype, rngs=rngs)
        self.out_proj = nnx.Linear(d_out, d_out, use_bias=False, dtype=dtype, rngs=rngs)

        # Direct JAX computation without caching
        cos, sin = precompute_rope_params(
            head_dim=self.head_dim,
            theta_base=rope_base,
            context_length=context_length,
            freq_config=rope_config
        )
        self.cos = cos.astype(dtype)
        self.sin = sin.astype(dtype)
        self.rope_base = rope_base
        self.freq_config = rope_config
        self.context_length = context_length

    def __call__(self, x: jax.Array) -> jax.Array:
        b, seq_len, _ = x.shape

        q = self.W_query(x).reshape(b, seq_len, self.num_heads, self.head_dim)
        k = self.W_key(x).reshape(b, seq_len, self.num_kv_groups, self.head_dim)
        v = self.W_value(x).reshape(b, seq_len, self.num_kv_groups, self.head_dim)

        # Transpose for attention shape: (b, heads, seq, dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # cos, sin = precompute_rope_params(
        #     head_dim=self.head_dim,
        #     theta_base=self.rope_base,
        #     context_length=seq_len,  # not the full context
        #     freq_config=self.freq_config
        # )
        # cos = cos.astype(x.dtype)
        # sin = sin.astype(x.dtype)
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        # Apply RoPE
        q = compute_rope(q, cos, sin)
        k = compute_rope(k, cos, sin)

        # Expand k and v from kv_groups to full head count
        k = jnp.repeat(k, self.group_size, axis=1)
        v = jnp.repeat(v, self.group_size, axis=1)

        # Attention scores
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q, k)  # (b, heads, query_len, key_len)

        # Apply causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        # attn_scores = jnp.where(mask, -jnp.inf, attn_scores)
        attn_scores = jnp.where(mask, attn_scores, -jnp.inf)


        # Softmax
        attn_weights = jax.nn.softmax(attn_scores / jnp.sqrt(self.head_dim), axis=-1)

        # Attention output
        context = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        context = jnp.transpose(context, (0, 2, 1, 3))  # (b, seq, heads, dim)
        context = context.reshape(b, seq_len, self.d_out)

        return self.out_proj(context)

class TransformerBlock(nnx.Module):
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        super().__init__()


        self.att = GroupedQueryAttention(
            d_in=args.embedding_dim,
            d_out=args.embedding_dim,
            context_length=args.context_length,
            num_heads=args.n_heads,
            num_kv_groups=args.n_kv_groups,
            rope_base=args.rope_base,
            rope_config=args.rope_freq,
            dtype=args.dtype,
            rngs=rngs
        )
        self.ff = FeedForward(args, rngs=rngs)
        self.norm1 = nnx.RMSNorm(args.embedding_dim, epsilon=1e-5, rngs=rngs)
        self.norm2 = nnx.RMSNorm(args.embedding_dim, epsilon=1e-5, rngs=rngs)

    def __call__(self, x, cos=None, sin=None, cache=None, cache_index=None):
        shortcut = x
        x = self.norm1(x)

        x = checkpoint(self.att)(x.astype(jnp.bfloat16))
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = checkpoint(self.ff)(x.astype(jnp.bfloat16))
        x = x + shortcut
        return x


class Llama3Model(nnx.Module):
    def __init__(self, args: ModelArgs, rngs: nnx.Rngs):
        super().__init__()

        self.tok_emb = nnx.Embed(
            num_embeddings=args.vocab_size,
            features=args.embedding_dim,
            dtype=args.dtype,
            param_dtype=args.param_dtype,
            rngs=rngs
        )

        self.trf_blocks = [
            TransformerBlock(args, rngs=rngs)
            for _ in range(args.n_layers)
        ]

        self.final_norm = nnx.RMSNorm(
            args.embedding_dim,
            epsilon=1e-5,
            dtype=args.dtype,
            param_dtype=args.param_dtype,
            rngs=rngs
        )

        self.out_head = nnx.Linear(
            args.embedding_dim,
            args.vocab_size,
            use_bias=False,
            dtype=args.dtype,
            param_dtype=args.param_dtype,
            rngs=rngs
        )

    def __call__(self, in_idx: jax.Array):
        x = self.tok_emb(in_idx)
        for block in self.trf_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.out_head(x.astype(jnp.bfloat16))
        return logits