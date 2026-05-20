# utils.py

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_best_heads(file_path, top_k=10):
    """Read the Voita results CSV and return list of (layer, head)."""
    if not os.path.exists(file_path):
        print(f"[WARN] Result file not found: {file_path}. Returning empty list.")
        return []
    try:
        df = pd.read_csv(file_path)
        # Sort by accuracy desc
        if "accuracy" in df.columns:
            df = df.sort_values("accuracy", ascending=False)
        top = df.head(top_k)
        return list(zip(top["layer"], top["head"]))
    except Exception as e:
        print(f"[WARN] Failed to read best heads from {file_path}: {e}")
        return []

def find_word_indices(sentence, target_word_idx, tokenizer):
    """
    Finds the token indices corresponding to the word at `target_word_idx` (0-based)
    in the space-separated sentence.
    """
    words = sentence.split()
    if target_word_idx >= len(words) or target_word_idx < 0:
        return []
    
    # Calculate char start/end for the target word
    char_start = 0
    for i in range(target_word_idx):
        char_start += len(words[i]) + 1 # +1 for space
    
    char_end = char_start + len(words[target_word_idx])
    
    # Tokenize and get offsets
    try:
        encoding = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    except:
        encoding = tokenizer(sentence, return_offsets_mapping=True)

    offsets = encoding["offset_mapping"]
    
    target_tokens = []
    for idx, (start, end) in enumerate(offsets):
        # Check for overlap: token start < word end AND token end > word start
        if start == end: continue 
        
        if start < char_end and end > char_start:
            target_tokens.append(idx)
            
    return target_tokens

def load_model_and_tokenizer(model_name, model_type):
    """
    Load tokenizer and model.
    model_type: "bert" or "gpt2"
    """
    print(f"Loading {model_type}: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if model_type == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True).to(DEVICE)
    else: # bert
        try:
            from transformers import AutoModelForMaskedLM
            model = AutoModelForMaskedLM.from_pretrained(model_name, output_attentions=True).to(DEVICE)
        except ImportError:
            print("    [WARN] AutoModelForMaskedLM not found, using AutoModel")
            model = AutoModel.from_pretrained(model_name, output_attentions=True).to(DEVICE)
            
    model.eval()
    return model, tokenizer
