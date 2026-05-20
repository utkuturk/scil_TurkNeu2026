# turkish_attention/analyze_surprisal.py

import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import find_word_indices, load_model_and_tokenizer, DEVICE

MODELS = {
    "bert": "dbmdz/bert-base-turkish-128k-cased",
    "gpt2": "redrussianarmy/gpt2-turkish-cased"
}

# Robust path handling
# BASE_DIR points to llm_code/, PROJECT_ROOT points to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STIMULI_PATH = os.path.join(PROJECT_ROOT, "stimuli", "tr_rus_all_conditions.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def calculate_surprisal(model, tokenizer, item, model_type):
    sentence = item["sentence"]
    
    # --- TURKISH INDEXING ---
    # Turkish is Head-Final (SOV). The verb is the last word.
    verb_word_idx = len(sentence.split()) - 1
    verb_tokens = find_word_indices(sentence, verb_word_idx, tokenizer)
    
    if not verb_tokens:
        return None

    # Prepare inputs
    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    original_input_ids = inputs["input_ids"].clone()
    
    surprisal = 0.0
    
    if model_type == "gpt2":
        # GPT-2 is Autoregressive (Causal). It cannot see the future.
        # No masking needed. We just check P(token | history).
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        for t_idx in verb_tokens:
            if t_idx > 0:
                 # Align: prediction for t_idx is at logits index t_idx-1
                 token_logit = shift_logits[0, t_idx-1, :]
                 token_id = shift_labels[0, t_idx-1]
                 log_prob = F.log_softmax(token_logit, dim=-1)[token_id].item()
                 surprisal += -log_prob
                 
    else: 
        # --- BERT (MASKED LM) CORRECTION ---
        # BERT is Bidirectional. It sees the target word if we don't hide it.
        # We must replace the verb tokens with [MASK] to measure "surprise".
        
        # We mask ALL verb tokens simultaneously (Whole Word Masking logic)
        # to see how much the context alone predicts the verb.
        masked_inputs = inputs["input_ids"].clone()
        
        # Get the mask token ID for this specific tokenizer
        mask_token_id = tokenizer.mask_token_id
        
        # Apply mask to all tokens belonging to the verb
        for t_idx in verb_tokens:
            masked_inputs[0, t_idx] = mask_token_id
            
        # Run model on MASKED sequence
        with torch.no_grad():
            outputs = model(input_ids=masked_inputs, attention_mask=inputs["attention_mask"])
        
        logits = outputs.logits
        
        # Calculate surprisal for the ORIGINAL tokens based on the MASKED context
        for t_idx in verb_tokens:
            token_logit = logits[0, t_idx, :]
            token_id = original_input_ids[0, t_idx] # The actual word that was there
            log_prob = F.log_softmax(token_logit, dim=-1)[token_id].item()
            surprisal += -log_prob
            
    return surprisal

def main():
    print("Running Turkish Surprisal Analysis...")
    if not os.path.exists(STIMULI_PATH):
        print(f"Stimuli file not found: {STIMULI_PATH}")
        return

    df = pd.read_csv(STIMULI_PATH)
    # Filter for Turkish
    df = df[df["lg"] == "turkish"]
    
    results = []
    
    for model_type, model_name in MODELS.items():
        print(f"Loading {model_type} ({model_name})...")
        model, tokenizer = load_model_and_tokenizer(model_name, model_type)
        model.eval()
        
        for _, row in df.iterrows():
            item = {"sentence": row["sentence"]}
            surprisal = calculate_surprisal(model, tokenizer, item, model_type)
            
            res_row = row.to_dict()
            res_row["model_type"] = model_type
            res_row["surprisal"] = surprisal
            results.append(res_row)
            
    if results:
        out_path = os.path.join(RESULTS_DIR, "turkish_surprisal_results.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()