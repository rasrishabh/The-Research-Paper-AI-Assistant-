import os
from transformers import AutoTokenizer
import streamlit as st

def token_size(input_text):
    # Load the Hugging Face token from the environment variables
    huggingface_token = st.secrets["general"]["HUGGINGFACE_TOKEN"]

    # Load the pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=huggingface_token)

    tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=0,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
        padding='max_length'
    )['input_ids'].flatten().tolist()

    return len(tokens) + 40
