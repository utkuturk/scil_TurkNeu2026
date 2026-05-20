# german_attention/analyze_surprisal.py

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

def calculate_surprisal(model, tokenizer, item, model_type):
    sentence = normalize_text(item["sentence"])
    words = sentence.split()
    
    # --- GERMAN VERB SEARCH ---
    # Your heuristic is correct for the dataset ("ist" / "sind").
    # If you ever add past tense, you might want to add "war"/"waren".
    verb_word_idx = -1
    for i, w in enumerate(words):
        if w.lower() in ["ist", "sind", "war", "waren"]:
            verb_word_idx = i
            break
            
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
        # No masking needed; it predicts next token from history.
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
        # FIX: We MUST mask the verb to measure how "surprised" BERT is.
        # Otherwise, BERT sees the word "ist" and predicts "ist" with ~100% confidence.
        
        masked_inputs = inputs["input_ids"].clone()
        mask_token_id = tokenizer.mask_token_id
        
        # Mask ALL tokens corresponding to the verb
        for t_idx in verb_tokens:
            masked_inputs[0, t_idx] = mask_token_id
            
        with torch.no_grad():
            outputs = model(input_ids=masked_inputs, attention_mask=inputs["attention_mask"])
            
        logits = outputs.logits
        
        # Calculate surprisal for the ORIGINAL token at the MASKED position
        for t_idx in verb_tokens:
            token_logit = logits[0, t_idx, :]
            token_id = original_input_ids[0, t_idx]
            log_prob = F.log_softmax(token_logit, dim=-1)[token_id].item()
            surprisal += -log_prob
            
    return surprisal

def main():
    print("Running German Surprisal Analysis...")
    if not os.path.exists(STIMULI_PATH):
        print(f"Stimuli file not found: {STIMULI_PATH}")
        return

    df = pd.read_csv(STIMULI_PATH)
    results = []
    
    for model_type, model_name in MODELS.items():
        print(f"Loading {model_type}: {model_name}")
        try:
            model, tokenizer = load_model_and_tokenizer(model_name, model_type)
            # Ensure model is in eval mode for consistent behavior
            model.eval()
            
            for _, row in df.iterrows():
                item = {
                    "sentence": row["sentence"]
                }
                surprisal = calculate_surprisal(model, tokenizer, item, model_type)
                
                res_row = row.to_dict()
                res_row["model_type"] = model_type
                res_row["surprisal"] = surprisal
                results.append(res_row)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    if results:
        out_path = os.path.join(RESULTS_DIR, "german_surprisal_results.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()