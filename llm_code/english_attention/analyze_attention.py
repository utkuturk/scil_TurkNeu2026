# english_attention/analyze_attention.py

import os
import sys
import pandas as pd
import numpy as np
import torch
import unicodedata
from scipy.stats import entropy

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_best_heads, find_word_indices, load_model_and_tokenizer, DEVICE

MODELS = {
    "bert": "bert-base-uncased",
    "gpt2": "gpt2"
}

# Robust path handling
# BASE_DIR points to llm_code/, PROJECT_ROOT points to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
NO_SYN_PATH = os.path.join(PROJECT_ROOT, "stimuli", "eng_no_syn_stimuli.csv")
SYN_PATH = os.path.join(PROJECT_ROOT, "stimuli", "eng_syn_stimuli.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def normalize_text(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return text
    
def calculate_attention(model, tokenizer, item, best_heads):
    sentence = normalize_text(item["sentence"])
    
    words = sentence.split()
    # Need at least 6 words for "The N P The N V" structure
    if len(words) < 6:
        return None
        
    # --- 1. HEAD NOUN ---
    # "The [Head]..." -> Always Index 1 (2nd word)
    head_tokens = find_word_indices(sentence, 1, tokenizer)

    # --- 2. ATTRACTOR ---
    # "The cake at the [senator's]..." -> Fixed at Index 4 (5th word) per your heuristics
    attractor_tokens = find_word_indices(sentence, 4, tokenizer)
    
    # --- 3. VERB ---
    # Dynamic search: The verb position changes (index 5 or 6).
    # We search for auxiliaries strictly AFTER the attractor (index > 4).
    verb_word_idx = -1
    for i, w in enumerate(words):
        if i > 4 and w.lower() in ["is", "are", "was", "were"]:
            verb_word_idx = i
            break
            
    if verb_word_idx == -1:
        # Fallback: If dynamic search fails, default to 5 
        # (Assuming the shorter structure: "The slogan on the poster is...")
        verb_word_idx = 5
        
    verb_tokens = find_word_indices(sentence, verb_word_idx, tokenizer)

    # Validate that we actually found tokens for all three positions
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

def process_file(path, model, tokenizer, best_heads, model_type, tag):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []
        
    df = pd.read_csv(path)
    results = []
    
    for _, row in df.iterrows():
        item = {
            "sentence": row["Full_Sentence"] if "Full_Sentence" in row else row.get("sentence", "")
        }
        
        metrics = calculate_attention(model, tokenizer, item, best_heads)
        
        res_row = row.to_dict()
        res_row["model_type"] = model_type
        res_row["source_file"] = tag
        if metrics:
            res_row.update(metrics)
        results.append(res_row)
        
    return results

def main():
    print("Running English Attention Analysis...")
    
    all_results = []
    
    for model_type, model_name in MODELS.items():
        model, tokenizer = load_model_and_tokenizer(model_name, model_type)
        best_heads_path = os.path.join(RESULTS_DIR, f"{model_type}_voita_head_accuracy_full.csv")
        best_heads = get_best_heads(best_heads_path)
        
        # Process No Syn
        all_results.extend(process_file(NO_SYN_PATH, model, tokenizer, best_heads, model_type, "no_syn"))
        
        # Process Syn
        all_results.extend(process_file(SYN_PATH, model, tokenizer, best_heads, model_type, "syn"))
            
    if all_results:
        out_path = os.path.join(RESULTS_DIR, "english_attention_results.csv")
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
