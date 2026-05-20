# russian_attention/analyze_attention.py

import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import entropy

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_best_heads, find_word_indices, load_model_and_tokenizer, DEVICE

MODELS = {
    "bert": "deepvk/bert-base-uncased",
    "gpt2": "ai-forever/rugpt3small_based_on_gpt2"
}

# Robust path handling
# BASE_DIR points to llm_code/, PROJECT_ROOT points to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STIMULI_PATH = os.path.join(PROJECT_ROOT, "stimuli", "tr_rus_all_conditions.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def calculate_attention(model, tokenizer, item, best_heads):
    # Ensure text is normalized if your tokenizer expects lowercase
    sentence = item["sentence"] 
    
    words = sentence.split()
    if len(words) < 4:
        return None

    # --- 1. HEAD ---
    # Always the first word (Index 0)
    head_tokens = find_word_indices(sentence, 0, tokenizer)

    # --- 2. ATTRACTOR ---
    # Always the third word (Index 2)
    # Structure: [Noun] [Prep] [Noun-Attractor]
    attractor_tokens = find_word_indices(sentence, 2, tokenizer)

    # --- 3. VERB SEARCH ---
    # The verb is separated from the attractor by a variable-length phrase.
    # In this dataset, the verb is always a form of "to be" (was/were):
    # был (masc), была (fem), было (neut), были (pl)
    verb_targets = ["был", "была", "было", "были"]
    
    verb_word_idx = -1
    for i, w in enumerate(words):
        # Start searching AFTER the attractor (index > 2)
        if i > 2 and w.lower() in verb_targets:
            verb_word_idx = i
            break
    
    # If not found, return None (skips this item)
    if verb_word_idx == -1:
        return None
        
    verb_tokens = find_word_indices(sentence, verb_word_idx, tokenizer)

    # --- VALIDATION ---
    if not head_tokens or not attractor_tokens or not verb_tokens:
        return None

    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions
    
    total_entropy = 0.0
    attn_to_head_sum = 0.0
    attn_to_attr_sum = 0.0
    count_heads = 0
    
    for layer, head in best_heads:
        if layer >= len(attentions): continue
        attn_matrix = attentions[layer][0, head] # [seq, seq]
        
        for v_idx in verb_tokens:
            dist = attn_matrix[v_idx]
            dist_np = dist.cpu().numpy()
            total_entropy += entropy(dist_np)
            count_heads += 1
            
            p_head = dist[head_tokens].sum().item()
            p_attr = dist[attractor_tokens].sum().item()
            
            attn_to_head_sum += p_head
            attn_to_attr_sum += p_attr
            
    if count_heads > 0:
        return {
            "attention_entropy": total_entropy / count_heads,
            "attention_diff": (attn_to_head_sum / count_heads) - (attn_to_attr_sum / count_heads)
        }
    return None

def main():
    print("Running Russian Attention Analysis...")
    if not os.path.exists(STIMULI_PATH):
        print(f"Stimuli file not found: {STIMULI_PATH}")
        return

    df = pd.read_csv(STIMULI_PATH)
    df = df[df["lg"] == "russian"]
    
    results = []
    
    for model_type, model_name in MODELS.items():
        model, tokenizer = load_model_and_tokenizer(model_name, model_type)
        best_heads_path = os.path.join(RESULTS_DIR, f"{model_type}_voita_head_accuracy_full.csv")
        best_heads = get_best_heads(best_heads_path)
        
        for _, row in df.iterrows():
            item = {
                "sentence": row["sentence"],
                "condition": row["condition"],
                "item": row["item"]
            }
            metrics = calculate_attention(model, tokenizer, item, best_heads)
            
            res_row = row.to_dict()
            res_row["model_type"] = model_type
            if metrics:
                res_row.update(metrics)
            results.append(res_row)
            
    if results:
        out_path = os.path.join(RESULTS_DIR, "russian_attention_results.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
