# voita_multihead.py
#
# Alternative BERT analysis methods for higher accuracy:
#
# 1. MULTI-HEAD AGGREGATION: Average attention across top-K heads or all heads per layer
# 2. MAX-POOLING: Take max attention across heads instead of single head
# 3. ATTENTION RANK: Check if target is in top-K attended tokens (not just argmax)
# 4. LAYER-WISE ANALYSIS: Aggregate across all heads within each layer
#
# These methods address the limitation that BERT's bidirectional attention
# distributes across many tokens, making single-head argmax ineffective.

import os
import argparse
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configurations per language
MODELS = {
    "english": "bert-base-uncased",
    "german": "bert-base-german-cased",
    "russian": "DeepPavlov/rubert-base-cased",
    "turkish": "dbmdz/bert-base-turkish-128k-cased",
}

# Dataset paths per language
DATASETS = {
    "english": "data/nsubj_root_sentences_en.csv",
    "german": "data/nsubj_root_sentences_de.csv",
    "russian": "data/nsubj_root_sentences_ru.csv",
    "turkish": "data/nsubj_root_sentences_tr.csv",
}


def find_wordpiece_span_bert(word: str, tokenizer, input_ids):
    """Find the start/end token indices for `word` inside `input_ids`."""
    word_ids = tokenizer.encode(word, add_special_tokens=False)
    if not word_ids:
        return None

    sent_ids = input_ids[0].tolist()
    max_start = len(sent_ids) - len(word_ids)

    for start in range(max_start + 1):
        if sent_ids[start:start + len(word_ids)] == word_ids:
            end = start + len(word_ids) - 1
            return start, end

    return None


def compute_multihead_metrics(
    df: pd.DataFrame,
    model_name: str,
    lang: str,
    log_every: int = 1000,
    top_k_ranks: list = [1, 3, 5, 10],
) -> dict:
    """
    Compute multiple metrics for BERT attention analysis:

    1. Per-head accuracy (standard Voita, both directions)
    2. Layer-averaged accuracy (average attention across all heads in a layer)
    3. Max-pooled accuracy (max attention across all heads)
    4. Top-K rank accuracy (is target in top-K attended tokens?)
    5. Attention score to target (raw attention value, not just argmax)

    Returns dict of DataFrames for each metric type.
    """
    print(f"[MultiHead-{lang}] Loading tokenizer and model: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.to(DEVICE)
    model.eval()

    max_len = model.config.max_position_embeddings
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # Accumulators for different metrics
    # 1. Per-head accuracy (nsubj → root direction)
    head_correct_nsubj_root = torch.zeros(num_layers, num_heads, dtype=torch.long)
    # 2. Per-head accuracy (leftward direction)
    head_correct_leftward = torch.zeros(num_layers, num_heads, dtype=torch.long)
    # 3. Layer-averaged accuracy
    layer_avg_correct = torch.zeros(num_layers, dtype=torch.long)
    # 4. Max-pooled across heads accuracy
    layer_max_correct = torch.zeros(num_layers, dtype=torch.long)
    # 5. Top-K rank accuracy per layer
    topk_correct = {k: torch.zeros(num_layers, dtype=torch.long) for k in top_k_ranks}
    # 6. Attention scores to target (for analysis)
    attention_scores_nsubj_root = []
    attention_scores_leftward = []

    total_processed = torch.zeros(1, dtype=torch.long)

    total_rows = len(df)
    print(f"[MultiHead-{lang}] Total rows: {total_rows} (max_len={max_len})")

    skipped_too_long = 0
    skipped_no_span = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        sentence = row["sentence"]
        nsubj = row["nsubj_first"]
        root = row["root_first"]

        if pd.isna(nsubj) or pd.isna(root):
            skipped_no_span += 1
            continue

        inputs = tokenizer(sentence, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        if seq_len > max_len:
            skipped_too_long += 1
            continue

        nsubj_span = find_wordpiece_span_bert(nsubj, tokenizer, inputs["input_ids"])
        root_span = find_wordpiece_span_bert(root, tokenizer, inputs["input_ids"])

        if nsubj_span is None or root_span is None:
            skipped_no_span += 1
            continue

        nsubj_idx = nsubj_span[0]
        root_idx = root_span[0]

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attn_tensor = torch.stack(outputs.attentions).squeeze(1)  # (L, H, seq, seq)

        # === Direction 1: nsubj → root (standard linguistic convention) ===
        source_nsubj_root = nsubj_idx
        target_nsubj_root = root_idx

        # === Direction 2: Leftward (source = rightmost, target = leftmost) ===
        if root_idx > nsubj_idx:
            source_leftward, target_leftward = root_idx, nsubj_idx
        else:
            source_leftward, target_leftward = nsubj_idx, root_idx

        # --- Per-head metrics ---
        # nsubj → root direction
        head_attn_nsubj_root = attn_tensor[:, :, source_nsubj_root, :]  # (L, H, seq)
        j_star_nsubj_root = head_attn_nsubj_root.argmax(dim=-1)  # (L, H)
        head_correct_nsubj_root += (j_star_nsubj_root == target_nsubj_root).cpu().long()

        # Leftward direction
        head_attn_leftward = attn_tensor[:, :, source_leftward, :]  # (L, H, seq)
        j_star_leftward = head_attn_leftward.argmax(dim=-1)  # (L, H)
        head_correct_leftward += (j_star_leftward == target_leftward).cpu().long()

        # --- Layer-averaged metrics (using leftward, which typically works better) ---
        # Average attention across all heads in each layer
        layer_avg_attn = head_attn_leftward.mean(dim=1)  # (L, seq)
        layer_avg_argmax = layer_avg_attn.argmax(dim=-1)  # (L,)
        layer_avg_correct += (layer_avg_argmax == target_leftward).cpu().long()

        # --- Max-pooled across heads ---
        layer_max_attn, _ = head_attn_leftward.max(dim=1)  # (L, seq)
        layer_max_argmax = layer_max_attn.argmax(dim=-1)  # (L,)
        layer_max_correct += (layer_max_argmax == target_leftward).cpu().long()

        # --- Top-K rank accuracy ---
        # Check if target is in top-K attended tokens (per layer, using layer-averaged attention)
        seq_len = layer_avg_attn.shape[-1]
        for k in top_k_ranks:
            # Handle case where sequence is shorter than k
            actual_k = min(k, seq_len)
            topk_indices = layer_avg_attn.topk(actual_k, dim=-1).indices  # (L, actual_k)
            in_topk = (topk_indices == target_leftward).any(dim=-1)  # (L,)
            topk_correct[k] += in_topk.cpu().long()

        # --- Store attention scores for later analysis ---
        # Store mean attention to target across all heads per layer
        attn_to_target_nsubj_root = head_attn_nsubj_root[:, :, target_nsubj_root].mean(dim=1)  # (L,)
        attn_to_target_leftward = head_attn_leftward[:, :, target_leftward].mean(dim=1)  # (L,)
        attention_scores_nsubj_root.append(attn_to_target_nsubj_root.cpu().numpy())
        attention_scores_leftward.append(attn_to_target_leftward.cpu().numpy())

        total_processed += 1

        if i % log_every == 0:
            n = total_processed.item()
            layer_avg_acc = (layer_avg_correct.float() / max(n, 1)).mean().item()
            head_leftward_acc = (head_correct_leftward.float() / max(n, 1)).mean().item()
            print(
                f"[MultiHead-{lang}] {i}/{total_rows} ({i/total_rows:.1%}) | "
                f"layer_avg_acc: {layer_avg_acc:.4f} | "
                f"head_leftward_acc: {head_leftward_acc:.4f} | "
                f"skipped: {skipped_too_long + skipped_no_span}",
                flush=True,
            )

    n = max(total_processed.item(), 1)
    print(f"[MultiHead-{lang}] Done. Processed: {total_processed.item()} | Skipped: {skipped_too_long + skipped_no_span}")

    # === Build result DataFrames ===
    results = {}

    # 1. Per-head accuracy (both directions)
    head_rows = []
    for layer in range(num_layers):
        for head in range(num_heads):
            head_rows.append({
                "language": lang,
                "layer": layer,
                "head": head,
                "direction": "nsubj_to_root",
                "correct": int(head_correct_nsubj_root[layer, head].item()),
                "total": n,
                "accuracy": float(head_correct_nsubj_root[layer, head].item()) / n,
            })
            head_rows.append({
                "language": lang,
                "layer": layer,
                "head": head,
                "direction": "leftward",
                "correct": int(head_correct_leftward[layer, head].item()),
                "total": n,
                "accuracy": float(head_correct_leftward[layer, head].item()) / n,
            })
    results["per_head"] = pd.DataFrame(head_rows)

    # 2. Layer-level metrics
    layer_rows = []
    for layer in range(num_layers):
        layer_rows.append({
            "language": lang,
            "layer": layer,
            "method": "layer_averaged",
            "correct": int(layer_avg_correct[layer].item()),
            "total": n,
            "accuracy": float(layer_avg_correct[layer].item()) / n,
        })
        layer_rows.append({
            "language": lang,
            "layer": layer,
            "method": "max_pooled",
            "correct": int(layer_max_correct[layer].item()),
            "total": n,
            "accuracy": float(layer_max_correct[layer].item()) / n,
        })
        for k in top_k_ranks:
            layer_rows.append({
                "language": lang,
                "layer": layer,
                "method": f"top_{k}_rank",
                "correct": int(topk_correct[k][layer].item()),
                "total": n,
                "accuracy": float(topk_correct[k][layer].item()) / n,
            })
    results["layer_metrics"] = pd.DataFrame(layer_rows)

    # 3. Attention scores summary
    attn_nsubj_root = np.array(attention_scores_nsubj_root)  # (n_samples, n_layers)
    attn_leftward = np.array(attention_scores_leftward)
    attn_rows = []
    for layer in range(num_layers):
        attn_rows.append({
            "language": lang,
            "layer": layer,
            "direction": "nsubj_to_root",
            "mean_attention": float(attn_nsubj_root[:, layer].mean()),
            "std_attention": float(attn_nsubj_root[:, layer].std()),
            "median_attention": float(np.median(attn_nsubj_root[:, layer])),
        })
        attn_rows.append({
            "language": lang,
            "layer": layer,
            "direction": "leftward",
            "mean_attention": float(attn_leftward[:, layer].mean()),
            "std_attention": float(attn_leftward[:, layer].std()),
            "median_attention": float(np.median(attn_leftward[:, layer])),
        })
    results["attention_scores"] = pd.DataFrame(attn_rows)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Multi-head BERT analysis with various aggregation methods."
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["english", "german", "russian", "turkish"],
        help="Language to process.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to nsubj-root dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/multihead",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N rows.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows to process (for testing).",
    )

    args = parser.parse_args()

    lang = args.language
    dataset_path = args.dataset_path or DATASETS[lang]
    model_name = MODELS[lang]
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"[MAIN] Language: {lang}")
    print(f"[MAIN] Model: {model_name}")
    print(f"[MAIN] Dataset: {dataset_path}")
    print(f"[MAIN] Output dir: {output_dir}")
    print(f"[MAIN] Device: {DEVICE}")

    df = pd.read_csv(dataset_path)
    print(f"[MAIN] Total dataset size: {len(df)} rows")

    if args.max_rows is not None and args.max_rows > 0:
        df = df.head(args.max_rows)
        print(f"[MAIN] Limiting to first {len(df)} rows for testing.")

    results = compute_multihead_metrics(df, model_name, lang, log_every=args.log_every)

    # Save all result DataFrames
    for name, result_df in results.items():
        output_path = os.path.join(output_dir, f"bert_multihead_{name}_{lang}.csv")
        result_df.to_csv(output_path, index=False)
        print(f"[MAIN] Saved {name} results to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print(f"SUMMARY FOR {lang.upper()}")
    print("="*60)

    # Best per-head (leftward)
    per_head_leftward = results["per_head"][results["per_head"]["direction"] == "leftward"]
    best_head = per_head_leftward.loc[per_head_leftward["accuracy"].idxmax()]
    print(f"\nBest single head (leftward): layer {int(best_head['layer'])}, head {int(best_head['head'])}")
    print(f"  Accuracy: {best_head['accuracy']:.4f} ({best_head['accuracy']*100:.2f}%)")

    # Best layer-averaged
    layer_avg = results["layer_metrics"][results["layer_metrics"]["method"] == "layer_averaged"]
    best_layer_avg = layer_avg.loc[layer_avg["accuracy"].idxmax()]
    print(f"\nBest layer-averaged: layer {int(best_layer_avg['layer'])}")
    print(f"  Accuracy: {best_layer_avg['accuracy']:.4f} ({best_layer_avg['accuracy']*100:.2f}%)")

    # Best max-pooled
    max_pooled = results["layer_metrics"][results["layer_metrics"]["method"] == "max_pooled"]
    best_max_pooled = max_pooled.loc[max_pooled["accuracy"].idxmax()]
    print(f"\nBest max-pooled: layer {int(best_max_pooled['layer'])}")
    print(f"  Accuracy: {best_max_pooled['accuracy']:.4f} ({best_max_pooled['accuracy']*100:.2f}%)")

    # Top-K rank accuracies
    print("\nTop-K rank accuracies (best layer):")
    for k in [1, 3, 5, 10]:
        topk = results["layer_metrics"][results["layer_metrics"]["method"] == f"top_{k}_rank"]
        best_topk = topk.loc[topk["accuracy"].idxmax()]
        print(f"  Top-{k}: layer {int(best_topk['layer'])}, accuracy {best_topk['accuracy']:.4f} ({best_topk['accuracy']*100:.2f}%)")


if __name__ == "__main__":
    main()
