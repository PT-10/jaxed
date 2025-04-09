# Code adapted from:
    # https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb
    # Licensed under the Apache License, Version 2.0
# Copyright 2023-2025 Sebastian Raschka


import os
import jax
import tiktoken
import jax.numpy as jnp
from pathlib import Path
from huggingface_hub import hf_hub_download
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)

        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens
        )


    def encode(self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()):
        if bos:
            tokens = [self.special_tokens["<|begin_of_text|>"]]
        else:
            tokens = []

        tokens += self.model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)


class ChatFormat:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode_header(self, message):
        tokens = []
        tokens.append(self.tokenizer.special_tokens["<|start_header_id|>"])
        tokens.extend(self.tokenizer.encode(message["role"], bos=False, eos=False))
        tokens.append(self.tokenizer.special_tokens["<|end_header_id|>"])
        tokens.extend(self.tokenizer.encode("\n\n", bos=False, eos=False))
        return tokens

    def encode(self, text):
        message = {
            "role": "user",
            "content": text
        }

        tokens = self.encode_header(message)
        tokens.extend(
            self.tokenizer.encode(message["content"].strip(), bos=False, eos=False)
        )
        tokens.append(self.tokenizer.special_tokens["<|eot_id|>"])
        return tokens

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = jnp.expand_dims(jnp.array(encoded),axis=0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """Generates text using the given model and parameters."""

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)  # Assuming model is already JAX-compatible
        logits = logits[:, -1, :]

        # Top-k sampling
        if top_k is not None:
            top_logits, _ = jax.lax.top_k(logits, top_k)
            min_val = top_logits[:, -1]
            logits = jnp.where(logits < min_val, -jnp.inf, logits)

        # Temperature scaling
        if temperature > 0.0:
            logits /= temperature
            probs = jax.nn.softmax(logits, axis=-1)
            idx_next = jax.random.categorical(jax.random.PRNGKey(0), probs, axis=-1, shape=(logits.shape[0], 1))
        else:
            idx_next = jnp.argmax(logits, axis=-1, keepdims=True)

        # Early stopping with eos_id
        if eos_id is not None and jnp.all(idx_next == eos_id):
            break

        idx = jnp.concatenate([idx, idx_next], axis=1)

    return idx


if __name__ == "__main__":
    # Assuming LLAMA_SIZE_STR is defined (e.g., "1B")
    LLAMA_SIZE_STR = "1B"

    tokenizer_file_path = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
        filename="original/tokenizer.model",
        local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    )

    tokenizer = Tokenizer(tokenizer_file_path)
    chat_tokenizer = ChatFormat(tokenizer)