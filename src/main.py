import yaml
from utils import *
from inference import *
import jax.numpy as jnp
from args import ModelArgs
from model import Llama3Model


if __name__ == "__main__":
    # Load the llama3 model config from the yaml file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Convert dtype and param_dtype to JAX types
    config["LLAMA32_CONFIG"]["dtype"] = jnp.bfloat16
    config["LLAMA32_CONFIG"]["param_dtype"] = jnp.bfloat16

    # Access the configuration
    LLAMA32_CONFIG = config["LLAMA32_CONFIG"]

    old_context_length = LLAMA32_CONFIG["context_length"]

    # Set new context length to reduce memory usage during inference
    LLAMA32_CONFIG["context_length"] = 8192

    LLAMA32_CONFIG["rope_base"] = rescale_theta(
        LLAMA32_CONFIG["rope_base"],
        old_context_length,
        LLAMA32_CONFIG["context_length"]
    )

    print("New RoPE theta:", LLAMA32_CONFIG["rope_base"])

    # Load the configs
    args = ModelArgs(**LLAMA32_CONFIG)

    # Instantiate model (params not allocated yet except ffn params since jax inits them)
    model = Llama3Model(args=args, rngs=nnx.Rngs(0))

    # Load weights from .safetensors file
    update_model_from_safetensors(model, "/content/Llama-3.2-1B-Instruct/model.safetensors")

    # Weight tying
    model.out_head.kernel = nnx.Param(model.tok_emb.embedding.value.T)

    # Load the tokenizer
    tokenizer = Tokenizer(tokenizer_file_path)
    chat_tokenizer = ChatFormat(tokenizer)

    # Default generation parameters
    max_new_tokens = 50
    top_k = 1
    temperature = 0.0

    # Print current parameters and ask if the user wants to change them
    print(f"Current generation parameters:")
    print(f"max_new_tokens: {max_new_tokens}, top_k: {top_k}, temperature: {temperature}")
    change_params = input("Do you want to change these parameters? (yes/no): ").strip().lower()

    if change_params == "yes":
        max_new_tokens = int(input("Enter new value for max_new_tokens: ").strip())
        top_k = int(input("Enter new value for top_k: ").strip())
        temperature = float(input("Enter new value for temperature: ").strip())

    # Continuous loop for user input
    while True:
        PROMPT = input("\nEnter your prompt (or type 'exit' to quit): ").strip()
        if PROMPT.lower() == "exit":
            print("Exiting the program.")
            break

        # Generate token IDs
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(PROMPT, chat_tokenizer),
            max_new_tokens=max_new_tokens,
            context_size=LLAMA32_CONFIG["context_length"],
            top_k=top_k,
            temperature=temperature
        )

        # Convert token IDs to text
        output_text = token_ids_to_text(token_ids, tokenizer)

        # Clean the output text
        cleaned_output = clean_output_text(output_text)

        # Print the cleaned output
        print("\nGenerated Output:")
        print(cleaned_output)