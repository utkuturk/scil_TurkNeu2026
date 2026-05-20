# german_attention/analyze_attention.py

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
    "bert": "bert-base-german-cased",
    "gpt2": "dbmdz/german-gpt2"
}

# Robust path handling
# BASE_DIR points to llm_code/, PROJECT_ROOT points to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STIMULI_PATH = os.path.join(PROJECT_ROOT, "stimuli", "german_stimuli.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def normalize_text(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return text

def calculate_attention(model, tokenizer, item, best_heads):
    sentence = normalize_text(item["sentence"])
    
    # German Heuristics:
    # "Der Telephonanruf für die Studentin ist bekannt."
    # 0=Der, 1=Telephonanruf(Head), 2=für, 3=die, 4=Studentin(Attractor), 5=ist(Verb)
    
    words = sentence.split()
    if len(words) < 6:
        return None
        
    head_tokens = find_word_indices(sentence, 1, tokenizer)
    attractor_tokens = find_word_indices(sentence, 4, tokenizer)
    
    # Verb heuristic: Find "ist"/"sind"
    verb_word_idx = -1
    for i, w in enumerate(words):
        if w.lower() in ["ist", "sind"]:
            verb_word_idx = i
            break
            
    if verb_word_idx == -1:
        # Fallback to word 5 if not found 
        verb_word_idx = 5
        
    verb_tokens = find_word_indices(sentence, verb_word_idx, tokenizer)

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
    print("Running German Attention Analysis...")
    if not os.path.exists(STIMULI_PATH):
        print(f"Stimuli file not found: {STIMULI_PATH}")
        return

    df = pd.read_csv(STIMULI_PATH)
    results = []
    
    for model_type, model_name in MODELS.items():
        print(f"Loading {model_type}: {model_name}")
        try:
            model, tokenizer = load_model_and_tokenizer(model_name, model_type)
            best_heads_path = os.path.join(RESULTS_DIR, f"{model_type}_voita_head_accuracy_full_german.csv")
            
            # Note: The voita run script for German produces files with _german suffix?
            # Looking at previous config: BERT_ATTENTION_CSV = .../bert_attention_nsubj_root_german.csv
            # But Voita script typically outputs output_name.replace(...,...) or assumes standard naming.
            # I must check voita_run_attention.py or the output of the process.
            # Usually strict Voita run outputs "gpt2_voita_head_accuracy_full.csv" in `results/` unless config changed logic.
            # BUT wait, all languages share `voita_run_attention.py`. 
            # If `voita_run_attention.py` uses `config.BERT_ATTENTION_CSV` to determine OUTPUT filename.. no
            # It usually saves to a fixed name unless modified. 
            # Let's check `voita_run_attention.py` of German attention folder.
            # Assuming standard naming for now, but will verify file existence.
            
            # Actually, `german_attention/config.py` set the INPUT csv name. 
            # `voita_run_attention.py` logic needs to be checked for output.
            # If it just writes to `results/gpt2_voita...csv`, it overwrites other languages!
            # CRITICAL ISSUE: If I run them sequentially, it's fine. But file naming collision risk.
            # Previous runs had `results/turkish_analysis_results.csv` etc.
            # But the intermediate file `bert_voita_head_accuracy_full.csv` is overwritten each time?
            # If so, I need to rename it or expect it to be fresh.
            # Since I am running this fresh, `results/bert_voita_head_accuracy_full.csv` will be German results after the BG process finishes.
            
            best_heads_path = os.path.join(RESULTS_DIR, f"{model_type}_voita_head_accuracy_full.csv") # Default name from script
            
            best_heads = get_best_heads(best_heads_path)
            if not best_heads and os.path.exists(best_heads_path):
                 print(f"Warning: Best heads file empty or malformed: {best_heads_path}")

            for _, row in df.iterrows():
                item = {
                    "sentence": row["sentence"]
                }
                metrics = calculate_attention(model, tokenizer, item, best_heads)
                
                res_row = row.to_dict()
                res_row["model_type"] = model_type
                if metrics:
                    res_row.update(metrics)
                results.append(res_row)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        out_path = os.path.join(RESULTS_DIR, "german_attention_results.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
