# english_attention/analyze_surprisal.py

import os
import sys
import pandas as pd
import torch
import unicodedata
import torch.nn.functional as F

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import find_word_indices, load_model_and_tokenizer, DEVICE

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

def calculate_surprisal(model, tokenizer, item, model_type):
    sentence = normalize_text(item["sentence"])
    words = sentence.split()
    
    # --- VERB SEARCH ---
    # Improved heuristic: Search for auxiliaries.
    # We search AFTER index 4 (start of attractor) to avoid false positives 
    # (e.g., if the subject phrase itself contained "is").
    verb_word_idx = -1
    verb_targets = ["is", "are", "was", "were"]
    
    for i, w in enumerate(words):
        if i > 4 and w.lower() in verb_targets:
            verb_word_idx = i
            break
            
    # Fallback to Index 5 (matches "The slogan on the poster is...")
    if verb_word_idx == -1:
        verb_word_idx = 5
    
    verb_tokens = find_word_indices(sentence, verb_word_idx, tokenizer)
    
    if not verb_tokens:
        return None

    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    original_input_ids = inputs["input_ids"].clone()
    
    surprisal = 0.0
    
    if model_type == "gpt2":
        # --- GPT-2 (Autoregressive) ---
        # No masking needed.
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        for t_idx in verb_tokens:
            if t_idx > 0:
                 token_logit = shift_logits[0, t_idx-1, :]
                 token_id = shift_labels[0, t_idx-1]
                 log_prob = F.log_softmax(token_logit, dim=-1)[token_id].item()
                 surprisal += -log_prob
                 
    else: 
        # --- BERT (Bidirectional) ---
        # FIX: We MUST mask the verb tokens to measure surprisal.
        masked_inputs = inputs["input_ids"].clone()
        mask_token_id = tokenizer.mask_token_id
        
        for t_idx in verb_tokens:
            masked_inputs[0, t_idx] = mask_token_id
            
        with torch.no_grad():
            outputs = model(input_ids=masked_inputs, attention_mask=inputs["attention_mask"])
            
        logits = outputs.logits
        
        # Calculate prob of ORIGINAL word at MASKED position
        for t_idx in verb_tokens:
            token_logit = logits[0, t_idx, :]
            token_id = original_input_ids[0, t_idx]
            log_prob = F.log_softmax(token_logit, dim=-1)[token_id].item()
            surprisal += -log_prob
            
    return surprisal

def process_file(path, model, tokenizer, model_type, tag):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []
        
    df = pd.read_csv(path)
    results = []
    
    for _, row in df.iterrows():
        # Handle column naming variations
        sent = row.get("Full_Sentence", row.get("sentence", ""))
        
        item = {
             "sentence": str(sent)
        }
        
        # Skip empty rows
        if not item["sentence"] or item["sentence"] == "nan":
            continue
            
        surprisal = calculate_surprisal(model, tokenizer, item, model_type)
        
        res_row = row.to_dict()
        res_row["model_type"] = model_type
        res_row["source_file"] = tag
        res_row["surprisal"] = surprisal
        results.append(res_row)
        
    return results

def main():
    print("Running English Surprisal Analysis...")
    
    all_results = []
    
    for model_type, model_name in MODELS.items():
        print(f"Loading {model_type} ({model_name})...")
        model, tokenizer = load_model_and_tokenizer(model_name, model_type)
        model.eval() # Ensure deterministic evaluation
        
        all_results.extend(process_file(NO_SYN_PATH, model, tokenizer, model_type, "no_syn"))
        all_results.extend(process_file(SYN_PATH, model, tokenizer, model_type, "syn"))
            
    if all_results:
        out_path = os.path.join(RESULTS_DIR, "english_surprisal_results.csv")
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()