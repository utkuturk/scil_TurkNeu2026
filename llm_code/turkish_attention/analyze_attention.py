import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import entropy

# Ensure we can import from parent directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_best_heads, find_word_indices, load_model_and_tokenizer, DEVICE

MODELS = {
    "bert": "dbmdz/bert-base-turkish-cased",
    "gpt2": "redrussianarmy/gpt2-turkish-cased"
}

# Robust path handling
# BASE_DIR points to llm_code/, PROJECT_ROOT points to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STIMULI_PATH = os.path.join(PROJECT_ROOT, "stimuli", "tr_rus_all_conditions.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def calculate_attention(model, tokenizer, item, best_heads):
    sentence = item["sentence"]
    
    # --- TURKISH INDICES SETUP ---
    # NOTE: You must ensure these indices match your Turkish stimuli structure.
    # If your stimuli CSV (tr_rus_all_conditions) has standard word ordering,
    # adjust these integers. 
    # Example for Subject-Attractor-Verb agreement usually involves:
    # 0: Head Noun (Subject)
    # 1 or 2: Attractor
    # Last: Verb
    
    # Using indices from your Russian code as placeholders:
    # 1. HEAD NOUN ("aşçısı" - The Cook)
    # This is at Index 1
    head_tokens = find_word_indices(sentence, 1, tokenizer)

    # 2. ATTRACTOR ("yöneticilerin" - The Managers')
    # This is at Index 0
    attractor_tokens = find_word_indices(sentence, 0, tokenizer)

    # 3. VERB ("zıpladılar" - Jumped)
    # This is at Index 4 (the last word)
    verb_tokens = find_word_indices(sentence, 4, tokenizer)
    
    if not head_tokens or not attractor_tokens or not verb_tokens:
        return None

    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions # Tuple of (batch, head, seq, seq)
    
    total_entropy = 0.0
    attn_to_head_sum = 0.0
    attn_to_attr_sum = 0.0
    count_heads = 0
    
    for layer, head in best_heads:
        # Safety check for layer index
        if layer >= len(attentions): 
            continue
            
        # Get attention matrix for this specific head [seq_len, seq_len]
        attn_matrix = attentions[layer][0, head] 
        
        # Check attention looking FROM the Verb
        for v_idx in verb_tokens:
            dist = attn_matrix[v_idx] # Attention distribution from verb token
            
            # Entropy calculation
            dist_np = dist.cpu().numpy()
            total_entropy += entropy(dist_np)
            count_heads += 1
            
            # Sum attention pointing TO head (subject) vs TO attractor
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
    print("Running Turkish Attention Analysis...")

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    if not os.path.exists(STIMULI_PATH):
        print(f"Stimuli file not found: {STIMULI_PATH}")
        return

    df = pd.read_csv(STIMULI_PATH)
    df = df[df["lg"] == "turkish"]
    print(f"Loaded {len(df)} Turkish sentences.")

    results = []

    for model_type, model_name in MODELS.items():
        print(f"Processing model: {model_name}")
        model, tokenizer = load_model_and_tokenizer(model_name, model_type)

        # Use Voita accuracy-based head selection (consistent with other languages)
        best_heads_path = os.path.join(RESULTS_DIR, f"{model_type}_voita_head_accuracy_full.csv")
        best_heads = get_best_heads(best_heads_path, top_k=10)

        if not best_heads:
            print(f"No best heads found for {model_type}. Run voita_run_attention.py first.")
            continue
        
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
            else:
                # Handle cases where tokens weren't found
                res_row["attention_entropy"] = None
                res_row["attention_diff"] = None
                
            results.append(res_row)
            
    if results:
        out_path = os.path.join(RESULTS_DIR, "turkish_attention_results.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()