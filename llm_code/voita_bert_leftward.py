# voita_bert_leftward.py
#
# BERT Voita-style head probing with LEFTWARD direction (root → nsubj)
# This is the original approach that yielded higher accuracy.
#
# The leftward constraint means: source = rightmost token, target = leftmost token
# This matches GPT-2's causal attention pattern and typically yields better results.

import os
import argparse
import pandas as pd
import torch
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
    """
    Find the start/end token indices for `word` inside `input_ids`
    for BERT (WordPiece). Returns (start, end) or None.
    """
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


def _log_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span, lang):
    total_clamped = total.clamp(min=1)
    mean_acc = (correct.float() / total_clamped).mean().item()
    progress_pct = i / total_rows if total_rows > 0 else 0

    print(
        f"[BERT-Leftward-{lang}] {i}/{total_rows} ({progress_pct:.1%}) | "
        f"mean_acc: {mean_acc:.4f} | "
        f"skipped (too long): {skipped_too_long} | "
        f"skipped (no span): {skipped_no_span}",
        flush=True,
    )


def compute_bert_leftward_voita(
    df: pd.DataFrame,
    model_name: str,
    lang: str,
    log_every: int = 1000
) -> pd.DataFrame:
    """
    Voita-style per-head accuracy for BERT with LEFTWARD direction.

    Direction: source = rightmost token, target = leftmost token
    This means if nsubj is before root, we do root → nsubj
    If root is before nsubj, we do nsubj → root
    """
    print(f"[BERT-Leftward-{lang}] Loading tokenizer and model: {model_name}...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.to(DEVICE)
    model.eval()

    max_len = model.config.max_position_embeddings
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    correct = torch.zeros(num_layers, num_heads, dtype=torch.long)
    total = torch.zeros(num_layers, num_heads, dtype=torch.long)

    total_rows = len(df)
    print(f"[BERT-Leftward-{lang}] Total rows: {total_rows} (max_len={max_len})")

    skipped_too_long = 0
    skipped_no_span = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        sentence = row["sentence"]
        nsubj = row["nsubj_first"]
        root = row["root_first"]

        if pd.isna(nsubj) or pd.isna(root):
            skipped_no_span += 1
            if i % log_every == 0:
                _log_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span, lang)
            continue

        inputs = tokenizer(sentence, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        if seq_len > max_len:
            skipped_too_long += 1
            if i % log_every == 0:
                _log_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span, lang)
            continue

        nsubj_span = find_wordpiece_span_bert(nsubj, tokenizer, inputs["input_ids"])
        root_span = find_wordpiece_span_bert(root, tokenizer, inputs["input_ids"])

        if nsubj_span is None or root_span is None:
            skipped_no_span += 1
            if i % log_every == 0:
                _log_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span, lang)
            continue

        nsubj_idx = nsubj_span[0]
        root_idx = root_span[0]

        # LEFTWARD direction: source = rightmost, target = leftmost
        # This is the original approach that yielded higher accuracy
        if root_idx > nsubj_idx:
            source_idx, target_idx = root_idx, nsubj_idx  # root → nsubj
        else:
            source_idx, target_idx = nsubj_idx, root_idx  # nsubj → root

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attn_tensor = torch.stack(outputs.attentions).squeeze(1)  # (L, H, seq, seq)

        # attention from source token to ALL tokens
        head_to_all = attn_tensor[:, :, source_idx, :]  # (L, H, seq)

        # argmax over j for each (layer, head)
        j_star = head_to_all.argmax(dim=-1)  # (L, H)
        is_correct = (j_star == target_idx)

        correct += is_correct.cpu().long()
        total += 1

        if i % log_every == 0:
            _log_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span, lang)

    print(
        f"[BERT-Leftward-{lang}] Done. Rows: {total_rows} | "
        f"skipped (too long): {skipped_too_long} | "
        f"skipped (no span): {skipped_no_span}"
    )

    total_clamped = total.clamp(min=1)
    accuracy = correct.float() / total_clamped

    rows = []
    for layer in range(num_layers):
        for head in range(num_heads):
            rows.append({
                "model": "bert",
                "direction": "leftward",
                "language": lang,
                "layer": layer,
                "head": head,
                "correct": int(correct[layer, head].item()),
                "total": int(total[layer, head].item()),
                "accuracy": float(accuracy[layer, head].item()),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="BERT Voita-style head accuracy with LEFTWARD direction (original approach)."
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
        help="Path to nsubj-root dataset CSV. If not specified, uses default for language.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/leftward",
        help="Output directory for results (default: results/leftward).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N rows (default: 1000).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (for testing).",
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

    result_df = compute_bert_leftward_voita(df, model_name, lang, log_every=args.log_every)

    output_path = os.path.join(output_dir, f"bert_leftward_{lang}.csv")
    result_df.to_csv(output_path, index=False)
    print(f"[MAIN] Saved results to {output_path}")

    print("\n[MAIN] Top 10 heads by accuracy:")
    print(result_df.sort_values("accuracy", ascending=False).head(10))


if __name__ == "__main__":
    main()
