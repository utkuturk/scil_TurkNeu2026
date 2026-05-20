#!/usr/bin/env python3
"""
Analyze stimuli attention using multiple head selection strategies:
1. Best leftward heads (from voita_bert_leftward.py results)
2. Best nsubj→root heads (from voita_multihead.py results)
3. Top-5 layer-aggregated attention (relaxed ranking method)

This script compares attention patterns across different probing methods
to see if they produce different patterns for agreement attraction stimuli.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from scipy.stats import entropy

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForMaskedLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Language-specific BERT models
BERT_MODELS = {
    "english": "bert-base-uncased",
    "german": "dbmdz/bert-base-german-cased",
    "russian": "DeepPavlov/rubert-base-cased",
    "turkish": "dbmdz/bert-base-turkish-cased"
}

# Stimuli file paths and language filters
STIMULI_CONFIG = {
    "english": {
        # "The slogan on the poster is designed..."
        # 0=The, 1=slogan(head), 2=on, 3=the, 4=poster(attractor), 5=is(verb)...
        "files": ["stimuli/eng_syn_stimuli.csv", "stimuli/eng_no_syn_stimuli.csv"],
        "filter_col": None,
        "sentence_col": "Full_Sentence",
        "word_indices": {"head": 1, "attractor": 4, "verb": 5}
    },
    "german": {
        # "Der Telephonanruf für die Studentin ist bekannt."
        # 0=Der, 1=Telephonanruf(head), 2=für, 3=die, 4=Studentin(attractor), 5=ist(verb)...
        "files": ["stimuli/german_stimuli.csv"],
        "filter_col": None,
        "sentence_col": "sentence",
        "word_indices": {"head": 1, "attractor": 4, "verb": 5}
    },
    "russian": {
        # Russian: "Абонемент на концерт был дорогим..."
        # 0=Абонемент(head), 1=на, 2=концерт(attractor), 3=был(verb)...
        "files": ["stimuli/tr_rus_all_conditions.csv"],
        "filter_col": ("lg", "russian"),
        "sentence_col": "sentence",
        "word_indices": {"head": 0, "attractor": 2, "verb": 3}
    },
    "turkish": {
        # Turkish: "yöneticilerin aşçısı mutfakta sürekli zıpladılar"
        # 0=attractor(genitive), 1=head(possessed), 2=location, 3=adverb, 4=verb
        "files": ["stimuli/tr_rus_all_conditions.csv"],
        "filter_col": ("lg", "turkish"),
        "sentence_col": "sentence",
        "word_indices": {"attractor": 0, "head": 1, "verb": 4}
    }
}


def load_best_heads(results_dir, language, method="leftward", top_k=10):
    """
    Load best heads from analysis results.

    method: "leftward" or "nsubj_to_root"
    """
    if method == "leftward":
        filepath = os.path.join(results_dir, "leftward", f"bert_leftward_{language}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df = df.sort_values("accuracy", ascending=False).head(top_k)
            return [(int(row["layer"]), int(row["head"])) for _, row in df.iterrows()]
    else:  # nsubj_to_root from multihead per-head results
        filepath = os.path.join(results_dir, "multihead", f"bert_multihead_per_head_{language}.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df = df[df["direction"] == "nsubj_to_root"]
            df = df.sort_values("accuracy", ascending=False).head(top_k)
            return [(int(row["layer"]), int(row["head"])) for _, row in df.iterrows()]

    print(f"[WARN] Could not load {method} heads for {language}")
    return []


def get_best_layer_for_top5(results_dir, language):
    """Get the layer with highest top-5 accuracy from multihead results."""
    filepath = os.path.join(results_dir, "multihead", f"bert_multihead_layer_metrics_{language}.csv")
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        top5 = df[df["method"] == "top_5_rank"]
        best = top5.loc[top5["accuracy"].idxmax()]
        return int(best["layer"])
    return 6  # Default to middle layer


def find_word_indices(sentence, target_word_idx, tokenizer):
    """Find token indices for word at position target_word_idx (supports negative indexing)."""
    words = sentence.split()

    # Handle negative indexing
    if target_word_idx < 0:
        target_word_idx = len(words) + target_word_idx

    if target_word_idx >= len(words) or target_word_idx < 0:
        return []

    # Calculate char start/end
    char_start = sum(len(words[i]) + 1 for i in range(target_word_idx))
    char_end = char_start + len(words[target_word_idx])

    # Tokenize with offsets
    try:
        encoding = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=True)
    except:
        encoding = tokenizer(sentence, return_offsets_mapping=True)

    offsets = encoding["offset_mapping"]

    target_tokens = []
    for idx, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if start < char_end and end > char_start:
            target_tokens.append(idx)

    return target_tokens


def calculate_attention_single_head(attentions, verb_tokens, head_tokens, attractor_tokens, heads):
    """
    Calculate attention metrics using specific heads.
    Returns entropy and attention_diff (head - attractor).
    """
    total_entropy = 0.0
    attn_to_head_sum = 0.0
    attn_to_attr_sum = 0.0
    count = 0

    for layer, head in heads:
        if layer >= len(attentions):
            continue

        attn_matrix = attentions[layer][0, head]  # [seq_len, seq_len]

        for v_idx in verb_tokens:
            dist = attn_matrix[v_idx]
            dist_np = dist.cpu().numpy()

            total_entropy += entropy(dist_np)
            attn_to_head_sum += dist[head_tokens].sum().item()
            attn_to_attr_sum += dist[attractor_tokens].sum().item()
            count += 1

    if count > 0:
        return {
            "entropy": total_entropy / count,
            "attn_diff": (attn_to_head_sum - attn_to_attr_sum) / count,
            "attn_to_head": attn_to_head_sum / count,
            "attn_to_attractor": attn_to_attr_sum / count
        }
    return None


def calculate_attention_top5_layer(attentions, verb_tokens, head_tokens, attractor_tokens, best_layer):
    """
    Calculate whether head noun is in top-5 attended tokens (aggregated across all heads in layer).
    Also returns attention metrics.
    """
    if best_layer >= len(attentions):
        return None

    # Average attention across all heads in the best layer
    layer_attn = attentions[best_layer][0].mean(dim=0)  # [seq_len, seq_len]

    total_entropy = 0.0
    attn_to_head_sum = 0.0
    attn_to_attr_sum = 0.0
    in_top5_count = 0
    count = 0

    for v_idx in verb_tokens:
        dist = layer_attn[v_idx]
        dist_np = dist.cpu().numpy()

        total_entropy += entropy(dist_np)
        attn_to_head_sum += dist[head_tokens].sum().item()
        attn_to_attr_sum += dist[attractor_tokens].sum().item()

        # Check if any head token is in top-5
        top5_indices = torch.topk(dist, min(5, len(dist))).indices.tolist()
        if any(h_idx in top5_indices for h_idx in head_tokens):
            in_top5_count += 1

        count += 1

    if count > 0:
        return {
            "entropy": total_entropy / count,
            "attn_diff": (attn_to_head_sum - attn_to_attr_sum) / count,
            "attn_to_head": attn_to_head_sum / count,
            "attn_to_attractor": attn_to_attr_sum / count,
            "head_in_top5": in_top5_count / count
        }
    return None


def process_language(language, results_dir, output_dir, top_k=10):
    """Process all stimuli for a single language."""
    print(f"\n{'='*60}")
    print(f"Processing {language.upper()}")
    print(f"{'='*60}")

    config = STIMULI_CONFIG[language]
    model_name = BERT_MODELS[language]

    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, output_attentions=True).to(DEVICE)
    model.eval()

    # Load stimuli
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    all_stimuli = []

    for stim_file in config["files"]:
        filepath = os.path.join(project_root, stim_file)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if config["filter_col"]:
                col, val = config["filter_col"]
                df = df[df[col] == val]
            df["source_file"] = os.path.basename(stim_file)
            all_stimuli.append(df)

    if not all_stimuli:
        print(f"No stimuli found for {language}")
        return

    stimuli_df = pd.concat(all_stimuli, ignore_index=True)
    print(f"Loaded {len(stimuli_df)} stimuli sentences")

    # Load best heads for different methods
    leftward_heads = load_best_heads(results_dir, language, "leftward", top_k)
    nsubj_heads = load_best_heads(results_dir, language, "nsubj_to_root", top_k)
    best_layer = get_best_layer_for_top5(results_dir, language)

    print(f"Leftward heads: {leftward_heads[:3]}... (top {len(leftward_heads)})")
    print(f"Nsubj→root heads: {nsubj_heads[:3]}... (top {len(nsubj_heads)})")
    print(f"Best top-5 layer: {best_layer}")

    word_indices = config["word_indices"]
    results = []

    # Determine sentence column from config
    sentence_col = config.get("sentence_col", "sentence")

    for idx, row in stimuli_df.iterrows():
        sentence = row[sentence_col]

        # Get token indices
        head_tokens = find_word_indices(sentence, word_indices["head"], tokenizer)
        attractor_tokens = find_word_indices(sentence, word_indices["attractor"], tokenizer)
        verb_tokens = find_word_indices(sentence, word_indices["verb"], tokenizer)

        if not head_tokens or not attractor_tokens or not verb_tokens:
            continue

        # Get attention matrices
        inputs = tokenizer(sentence, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        attentions = outputs.attentions

        result_row = row.to_dict()
        result_row["language"] = language

        # Method 1: Leftward heads
        if leftward_heads:
            metrics = calculate_attention_single_head(
                attentions, verb_tokens, head_tokens, attractor_tokens, leftward_heads
            )
            if metrics:
                result_row["leftward_entropy"] = metrics["entropy"]
                result_row["leftward_attn_diff"] = metrics["attn_diff"]
                result_row["leftward_attn_to_head"] = metrics["attn_to_head"]
                result_row["leftward_attn_to_attractor"] = metrics["attn_to_attractor"]

        # Method 2: Nsubj→root heads
        if nsubj_heads:
            metrics = calculate_attention_single_head(
                attentions, verb_tokens, head_tokens, attractor_tokens, nsubj_heads
            )
            if metrics:
                result_row["nsubj_entropy"] = metrics["entropy"]
                result_row["nsubj_attn_diff"] = metrics["attn_diff"]
                result_row["nsubj_attn_to_head"] = metrics["attn_to_head"]
                result_row["nsubj_attn_to_attractor"] = metrics["attn_to_attractor"]

        # Method 3: Top-5 layer aggregation
        metrics = calculate_attention_top5_layer(
            attentions, verb_tokens, head_tokens, attractor_tokens, best_layer
        )
        if metrics:
            result_row["top5_layer"] = best_layer
            result_row["top5_entropy"] = metrics["entropy"]
            result_row["top5_attn_diff"] = metrics["attn_diff"]
            result_row["top5_attn_to_head"] = metrics["attn_to_head"]
            result_row["top5_attn_to_attractor"] = metrics["attn_to_attractor"]
            result_row["top5_head_in_top5"] = metrics["head_in_top5"]

        results.append(result_row)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(stimuli_df)} sentences")

    # Save results
    if results:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{language}_multihead_stimuli_attention.csv")
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved {len(results)} results to {out_path}")

        # Print summary statistics
        df_results = pd.DataFrame(results)
        print(f"\n--- {language.upper()} Summary ---")
        for method in ["leftward", "nsubj", "top5"]:
            diff_col = f"{method}_attn_diff"
            if diff_col in df_results.columns:
                mean_diff = df_results[diff_col].mean()
                std_diff = df_results[diff_col].std()
                print(f"  {method}: attn_diff = {mean_diff:.4f} ± {std_diff:.4f}")

        if "top5_head_in_top5" in df_results.columns:
            print(f"  top5: head_in_top5 rate = {df_results['top5_head_in_top5'].mean():.2%}")


def main():
    parser = argparse.ArgumentParser(description="Analyze stimuli with multiple head selection methods")
    parser.add_argument("--language", "-l", type=str, default="all",
                        choices=["english", "german", "russian", "turkish", "all"],
                        help="Language to process (default: all)")
    parser.add_argument("--results-dir", "-r", type=str, default="results",
                        help="Directory containing leftward/multihead results")
    parser.add_argument("--output-dir", "-o", type=str, default="results/stimuli_multihead",
                        help="Output directory for results")
    parser.add_argument("--top-k", "-k", type=int, default=10,
                        help="Number of top heads to use")

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, args.results_dir)
    output_dir = os.path.join(project_root, args.output_dir)

    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {DEVICE}")

    languages = ["english", "german", "russian", "turkish"] if args.language == "all" else [args.language]

    for lang in languages:
        process_language(lang, results_dir, output_dir, args.top_k)

    print("\n" + "="*60)
    print("All done!")
    print("="*60)


if __name__ == "__main__":
    main()
