# russian_attention/analyze_surprisal.py

import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import find_word_indices, load_model_and_tokenizer, DEVICE

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

def find_russian_verb_index(sentence):
    """
    Dynamically finds the index of the main verb (forms of 'to be')
    starting after the attractor (index > 2).
    """
    words = sentence.split()
    verb_targets = ["был", "была", "было", "были"]
    
    # We look for the verb starting from index 3 to skip the attractor phrase start
    # Structure: [Head] [Prep] [Attractor] ... [Verb]
    for i, w in enumerate(words):
        if i > 2 and w.lower() in verb_targets:
            return i
    return -1

def calculate_surprisal(model, tokenizer, item, model_type):
    sentence = item["sentence"]

    # --- DYNAMIC VERB FINDING ---
    verb_word_idx = find_russian_verb_index(sentence)

    if verb_word_idx == -1:
        return None

    verb_tokens = find_word_indices(sentence, verb_word_idx, tokenizer)

    if not verb_tokens:
        return None

    inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
    original_input_ids = inputs["input_ids"].clone()

    surprisal = 0.0

    if model_type == "gpt2":
        # GPT-2 is autoregressive: predict token T given 0..T-1
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        for t_idx in verb_tokens:
            if t_idx > 0:
                 if t_idx - 1 < shift_logits.size(1):
                     token_logit = shift_logits[0, t_idx-1, :]
                     token_id = shift_labels[0, t_idx-1]
                     log_prob = F.log_softmax(token_logit, dim=-1)[token_id].item()
                     surprisal += -log_prob
    else:
        # --- BERT (Bidirectional) ---
        # PLL: Mask verb tokens, then compute P(original | masked context)
        masked_inputs = inputs["input_ids"].clone()
        mask_token_id = tokenizer.mask_token_id

        for t_idx in verb_tokens:
            masked_inputs[0, t_idx] = mask_token_id

        with torch.no_grad():
            outputs = model(input_ids=masked_inputs, attention_mask=inputs["attention_mask"])

        logits = outputs.logits

        for t_idx in verb_tokens:
            token_logit = logits[0, t_idx, :]
            token_id = original_input_ids[0, t_idx]
            log_prob = F.log_softmax(token_logit, dim=-1)[token_id].item()
            surprisal += -log_prob

    return surprisal

def main():
    print("Running Russian Surprisal Analysis...")
    if not os.path.exists(STIMULI_PATH):
        print(f"Stimuli file not found: {STIMULI_PATH}")
        return

    df = pd.read_csv(STIMULI_PATH)
    df = df[df["lg"] == "russian"]
    
    results = []
    
    for model_type, model_name in MODELS.items():
        print(f"Loading {model_type}...")
        model, tokenizer = load_model_and_tokenizer(model_name, model_type)
        model.eval() # Ensure eval mode
        
        for _, row in df.iterrows():
            item = {"sentence": row["sentence"]}
            surprisal = calculate_surprisal(model, tokenizer, item, model_type)
            
            res_row = row.to_dict()
            res_row["model_type"] = model_type
            res_row["surprisal"] = surprisal
            results.append(res_row)
            
    if results:
        out_path = os.path.join(RESULTS_DIR, "russian_surprisal_results.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()