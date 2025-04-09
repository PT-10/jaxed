import gc
from tqdm import tqdm
from flax import nnx
import jax.numpy as jnp
from safetensors.numpy import safe_open


def rescale_theta(theta_old, context_length_old, context_length_new):
    scaling_factor = context_length_new / context_length_old
    return theta_old * scaling_factor


def clean_output_text(output_text, special_tokens=None):
    if special_tokens is None:
        special_tokens = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]

    # Remove special tokens
    for token in special_tokens:
        output_text = output_text.replace(token, "")

    # Strip leading/trailing whitespace and normalize spaces
    output_text = " ".join(output_text.split())

    return output_text


def assign(dest, src, name):
    if dest.shape != src.shape:
        raise ValueError(
            f"[Shape Mismatch] → {name}\n"
            f"  Expected: {dest.shape}\n"
            f"  Received: {src.shape}\n"
        )
    # if dest.dtype != src.dtype:
    #     print(
    #         f"[Warning: Dtype Mismatch] → {name}\n"
    #         f"  Expected: {dest.dtype}\n"
    #         f"  Received: {src.dtype}\n"
    #     )
    return src


def update_model_from_safetensors(model, safetensors_path):
    with safe_open(safetensors_path, framework="np") as f:
        for key in tqdm(f.keys(), desc="Loading Weights"):
            tensor = jnp.array(f.get_tensor(key))

            if key == 'model.embed_tokens.weight':
                model.tok_emb.embedding.value = assign(model.tok_emb.embedding.value, tensor, key)

            elif key.startswith('model.layers.'):
                parts = key.split('.')
                layer_id = int(parts[2])
                subkey = '.'.join(parts[3:])
                block = model.trf_blocks[layer_id]
                # print(subkey)

                if subkey == 'self_attn.q_proj.weight':
                    block.att.W_query.kernel.value = assign(block.att.W_query.kernel.value, tensor.T, key)
                elif subkey == 'self_attn.k_proj.weight':
                    block.att.W_key.kernel.value = assign(block.att.W_key.kernel.value, tensor.T, key)
                elif subkey == 'self_attn.v_proj.weight':
                    block.att.W_value.kernel.value = assign(block.att.W_value.kernel.value, tensor.T, key)
                elif subkey == 'self_attn.o_proj.weight':
                    block.att.out_proj.kernel.value = assign(block.att.out_proj.kernel.value, tensor.T, key)
                elif subkey == 'input_layernorm.weight':
                    block.norm1.scale.value = assign(block.norm1.scale.value, tensor, key)
                elif subkey == 'post_attention_layernorm.weight':
                    block.norm2.scale.value = assign(block.norm2.scale.value, tensor, key)
                elif subkey == 'mlp.gate_proj.weight':
                    block.ff.fc1.kernel.value = assign(block.ff.fc1.kernel.value, tensor.T, key)
                elif subkey == 'mlp.up_proj.weight':
                    block.ff.fc2.kernel.value = assign(block.ff.fc2.kernel.value, tensor.T, key)
                elif subkey == 'mlp.down_proj.weight':
                    block.ff.fc3.kernel.value = assign(block.ff.fc3.kernel.value, tensor.T, key)

            elif key == 'model.norm.weight':
                model.final_norm.scale.value = assign(model.final_norm.scale.value, tensor, key)

            elif key == 'lm_head.weight':
                model.out_head.kernel.value = assign(model.out_head.kernel.value, tensor.T, key)

            del tensor
            gc.collect()


def get_all_params(module):
    params = []

    if isinstance(module, nnx.Param):
        params.append(module)
    elif isinstance(module, nnx.Module):
        for attr in vars(module).values():
            params.extend(get_all_params(attr))
    elif isinstance(module, (list, tuple)):
        for item in module:
            params.extend(get_all_params(item))
    elif isinstance(module, dict):
        for item in module.values():
            params.extend(get_all_params(item))

    return params


def count_params_and_size(model):
    params = get_all_params(model)
    total_params = sum(p.value.size for p in params)
    total_bytes = sum(p.value.size * p.value.dtype.itemsize for p in params)
    return total_params, total_bytes


def pretty_size(num_bytes):
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.2f} KB"
    elif num_bytes < 1024**3:
        return f"{num_bytes / (1024**2):.2f} MB"
    else:
        return f"{num_bytes / (1024**3):.2f} GB"