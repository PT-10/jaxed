import os
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download

# Load environment variables from .env file
load_dotenv()

# Check if the Hugging Face token is already stored in the .env file
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    # Perform interactive login to Hugging Face
    print("Please log in to Hugging Face:")
    HF_TOKEN = login()
    
    # Save the token to the .env file
    with open(".env", "a") as env_file:
        env_file.write(f"HF_TOKEN={HF_TOKEN}\n")
    print("Hugging Face token saved to .env file.")

# Set the Hugging Face token as an environment variable
os.environ["HF_TOKEN"] = HF_TOKEN

# Define the repository and filenames
REPO_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_FILENAME = "model.safetensors"
TOKENIZER_FILENAME = "original/tokenizer.model"

# Download the model weights
weights_file = hf_hub_download(
    repo_id=REPO_ID,
    filename=MODEL_FILENAME,
    local_dir="Llama-3.2-1B-Instruct",
    token=HF_TOKEN
)
print(f"Model weights downloaded to: {weights_file}")

# Download the tokenizer
tokenizer_file = hf_hub_download(
    repo_id=REPO_ID,
    filename=TOKENIZER_FILENAME,
    local_dir="Llama-3.2-1B-Instruct/original",
    token=HF_TOKEN
)
print(f"Tokenizer downloaded to: {tokenizer_file}")