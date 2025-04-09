# JAX - Llama-3.2-1B-Instruct Model

This repository contains the implementation of the **Llama-3.2-1B-Instruct** model using **JAX** and **Flax**. It includes model definition, loading pretrained weights, tokenizer integration, and inference for text generation.

---

## Table of Contents

1. [Overview](#overview)

2. [Requirements](#requirements)

3. [Usage](#usage)

7. [Acknowledgments](#acknowledgments)

---

## Overview

The **Llama-3.2-1B-Instruct** model is a decoder-only transformer-based language model finetuned for instruction-following tasks. This implementation demonstrates:

- Defining the model architecture using **Flax**.
- Loading pretrained weights from Hugging Face's `safetensors` and converting them to Jax Arrays.

- Performing inference to generate text.

---



## Usage

#### Clone the repository
```bash
git clone https://github.com/jaxed.git
cd jaxed/src
```

#### Download model weights and tokenizer file by running

- Before running the script, get a HF access token and place it in `.env` as **HF_TOKEN** in the root folder. 

```bash
python download_checkpoint.py
```

#### Running the Script
```bash
python main.py
```

- The script will display the current generation parameters (max_new_tokens, top_k, temperature) and ask if you want to modify them.
Enter your prompt, and the model will generate a response.
- Exit: Type exit to terminate the program.

## Requirements

To run the project, ensure the dependencies are installed.

- This project uses Python3.11


### Install Dependencies

Install the required libraries using the following command:

```bash
pip install -r src/requirements.txt
```

---

## Acknowledgments

**[Sebastian Raschka](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/07_gpt_to_llama/standalone-llama32.ipynb)**  for providing the PyTorch implementation.
