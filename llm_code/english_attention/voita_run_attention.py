# voita_run_attention.py
#
# One-pass Voita-style per-head accuracy for BERT and GPT2.
# Added optional argument --max-rows for testing a subset.

import os
import argparse
import pandas as pd
import torch
from transformers import (
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModel,
)

try:
    from .config import (
        BERT_MODEL_NAME,
        GPT2_MODEL_NAME,
        N_SUBJ_ROOT_CSV,
    )
except ImportError:
    from config import (
        BERT_MODEL_NAME,
        GPT2_MODEL_NAME,
        N_SUBJ_ROOT_CSV,
    )

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- helper: dataset ----------

def load_dataset(path: str = N_SUBJ_ROOT_CSV) -> pd.DataFrame:
    """Load the sentence / nsubj / root dataset."""
    return pd.read_csv(path)


# ---------- helper: subword alignment ----------

def find_wordpiece_span_bert(word: str, tokenizer, input_ids):
    """
    Find the start/end token indices for `word` inside `input_ids`
    for BERT (WordPiece). Returns (start, end) or None.
    """
    # Note: The NaN check must happen BEFORE calling this function, as 'word' must be a string.
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


def find_subtoken_span_gpt2(word: str, tokenizer, input_ids, sentence: str = None):
    """
    Find the start/end token indices for `word` inside `input_ids`
    for GPT-2 (BPE). Returns (start, end) or None.

    Uses character-level offset mapping to handle BPE's context-sensitive tokenization.
    """
    if sentence is None:
        # Fallback to old method (may fail due to BPE context sensitivity)
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

    # Use offset mapping for accurate alignment
    encoding = tokenizer(sentence, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding.get("offset_mapping", None)

    if offsets is None:
        return find_subtoken_span_gpt2(word, tokenizer, input_ids, sentence=None)

    # Find the word's character span in the sentence
    word_start_char = sentence.find(word)
    if word_start_char == -1:
        return None
    word_end_char = word_start_char + len(word)

    # Find which tokens overlap with this character span
    token_start = None
    token_end = None

    for idx, (start_char, end_char) in enumerate(offsets):
        if start_char < word_end_char and end_char > word_start_char:
            if token_start is None:
                token_start = idx
            token_end = idx

    if token_start is None:
        return None

    return token_start, token_end


# ---------- BERT Voita ----------

def _log_bert_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span):
    total_clamped = total.clamp(min=1)
    mean_acc = (correct.float() / total_clamped).mean().item()
    print(
        f"[BERT-Voita] {i}/{total_rows} rows | "
        f"mean head accuracy so far: {mean_acc:.4f} | "
        f"skipped (too long): {skipped_too_long} | "
        f"skipped (no span): {skipped_no_span}",
        flush=True,
    )


def compute_bert_voita_counts(df: pd.DataFrame, log_every: int = 1000) -> pd.DataFrame:
    """
    Voita-style per-head accuracy for BERT over the entire DataFrame.
    """
    print("[BERT-Voita] Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertModel.from_pretrained(BERT_MODEL_NAME, output_attentions=True)
    model.to(DEVICE)
    model.eval()

    max_len = model.config.max_position_embeddings
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    correct = torch.zeros(num_layers, num_heads, dtype=torch.long)
    total = torch.zeros(num_layers, num_heads, dtype=torch.long)

    total_rows = len(df)
    print(f"[BERT-Voita] Total rows: {total_rows} (max_len={max_len})")

    skipped_too_long = 0
    skipped_no_span = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        sentence = row["sentence"]
        nsubj = row["nsubj_first"]
        root = row["root_first"]

        # FIX: Check for NaN/missing values before tokenizing (Addresses the ValueError)
        if pd.isna(nsubj) or pd.isna(root):
            skipped_no_span += 1
            if i % log_every == 0:
                _log_bert_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span)
            continue
        
        inputs = tokenizer(sentence, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        if seq_len > max_len:
            skipped_too_long += 1
            if i % log_every == 0:
                _log_bert_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span)
            continue

        nsubj_span = find_wordpiece_span_bert(nsubj, tokenizer, inputs["input_ids"])
        root_span = find_wordpiece_span_bert(root, tokenizer, inputs["input_ids"])

        if nsubj_span is None or root_span is None:
            skipped_no_span += 1
            if i % log_every == 0:
                _log_bert_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span)
            continue

        nsubj_idx = nsubj_span[0]
        root_idx = root_span[0]

        # BERT direction: dependent -> head (nsubj -> root)
        # This is the standard linguistic convention for dependency probing
        source_idx, target_idx = nsubj_idx, root_idx

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attn_tensor = torch.stack(outputs.attentions).squeeze(1)  # (L, H, seq, seq)

        # attention from source token (nsubj) to ALL tokens
        head_to_all = attn_tensor[:, :, source_idx, :]  # (L, H, seq)

        # argmax over j for each (layer, head)
        j_star = head_to_all.argmax(dim=-1)  # (L, H)
        is_correct = (j_star == target_idx)

        correct += is_correct.cpu().long()
        total += 1  # one trial per head for this dependency instance

        if i % log_every == 0:
            _log_bert_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span)

    print(
        f"[BERT-Voita] Done. Rows: {total_rows} | "
        f"skipped (too long): {skipped_too_long} | "
        f"skipped (no span): {skipped_no_span}"
    )

    total_clamped = total.clamp(min=1)
    accuracy = correct.float() / total_clamped

    rows = []
    for layer in range(num_layers):
        for head in range(num_heads):
            rows.append(
                {
                    "model": "bert",
                    "layer": layer,
                    "head": head,
                    "correct": int(correct[layer, head].item()),
                    "total": int(total[layer, head].item()),
                    "accuracy": float(accuracy[layer, head].item()),
                }
            )

    return pd.DataFrame(rows)


# ---------- GPT2 Voita ----------

def _log_gpt2_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span):
    total_clamped = total.clamp(min=1)
    mean_acc = (correct.float() / total_clamped).mean().item()
    progress_pct = i / total_rows if total_rows > 0 else 0
    print(
        f"[GPT2-Voita] {i}/{total_rows} ({progress_pct:.1%}) | "
        f"skipped (too long): {skipped_too_long} | "
        f"skipped (no span): {skipped_no_span}",
        flush=True,
    )


def compute_gpt2_voita_counts(df: pd.DataFrame, log_every: int = 1000) -> pd.DataFrame:
    """
    Voita-style per-head accuracy for GPT2 over the entire DataFrame.
    """
    print(f"[GPT2-Voita] Loading tokenizer and model: {GPT2_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(GPT2_MODEL_NAME)
    model = AutoModel.from_pretrained(GPT2_MODEL_NAME, output_attentions=True)
    model.to(DEVICE)
    model.eval()

    max_len = model.config.n_positions
    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    correct = torch.zeros(num_layers, num_heads, dtype=torch.long)
    total = torch.zeros(num_layers, num_heads, dtype=torch.long)

    total_rows = len(df)
    print(f"[GPT2-Voita] Total rows: {total_rows} (max_len={max_len})")

    skipped_too_long = 0
    skipped_no_span = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        sentence = row["sentence"]
        nsubj = row["nsubj_first"]
        root = row["root_first"]

        if pd.isna(nsubj) or pd.isna(root):
            skipped_no_span += 1
            if i % log_every == 0:
                _log_gpt2_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span)
            continue

        inputs = tokenizer(sentence, return_tensors="pt")
        seq_len = inputs["input_ids"].shape[1]

        if seq_len > max_len:
            skipped_too_long += 1
            if i % log_every == 0:
                _log_gpt2_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span)
            continue

        nsubj_span = find_subtoken_span_gpt2(nsubj, tokenizer, inputs["input_ids"], sentence=sentence)
        root_span = find_subtoken_span_gpt2(root, tokenizer, inputs["input_ids"], sentence=sentence)

        if nsubj_span is None or root_span is None:
            skipped_no_span += 1
            if i % log_every == 0:
                _log_gpt2_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span)
            continue

        nsubj_idx = nsubj_span[0]
        root_idx = root_span[0]

        # GPT-2 direction: must be leftward (causal attention)
        # Source = later token, Target = earlier token
        if root_idx > nsubj_idx:
            source_idx, target_idx = root_idx, nsubj_idx
        else:
            source_idx, target_idx = nsubj_idx, root_idx

        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attn_tensor = torch.stack(outputs.attentions).squeeze(1)  # (L, H, seq, seq)

        head_to_all = attn_tensor[:, :, source_idx, :]  # (L, H, seq)

        j_star = head_to_all.argmax(dim=-1)  # (L, H)
        is_correct = (j_star == target_idx)

        correct += is_correct.cpu().long()
        total += 1

        if i % log_every == 0:
            _log_gpt2_voita_progress(i, total_rows, correct, total, skipped_too_long, skipped_no_span)

    print(
        f"[GPT2-Voita] Done. Rows: {total_rows} | "
        f"skipped (too long): {skipped_too_long} | "
        f"skipped (no span): {skipped_no_span}"
    )

    total_clamped = total.clamp(min=1)
    accuracy = correct.float() / total_clamped

    rows = []
    for layer in range(num_layers):
        for head in range(num_heads):
            rows.append(
                {
                    "model": "gpt2",
                    "layer": layer,
                    "head": head,
                    "correct": int(correct[layer, head].item()),
                    "total": int(total[layer, head].item()),
                    "accuracy": float(accuracy[layer, head].item()),
                }
            )

    return pd.DataFrame(rows)


# ---------- main ----------

def main(
    dataset_path: str = N_SUBJ_ROOT_CSV,
    bert_out: str = "results/bert_voita_head_accuracy_full.csv",
    gpt2_out: str = "results/gpt2_voita_head_accuracy_full.csv",
    log_every: int = 1000,
    max_rows: int = None,
    run_bert: bool = True,
    run_gpt2: bool = True,
):
    os.makedirs("results", exist_ok=True)

    print(f"[MAIN] Loading dataset from {dataset_path}")
    df = load_dataset(dataset_path)
    print(f"[MAIN] Total dataset size: {len(df)} rows")

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
        print(f"[MAIN] Limiting processing to first {len(df)} rows for test.")

    # BERT
    if run_bert:
        bert_df = compute_bert_voita_counts(df, log_every=log_every)
        bert_df.to_csv(bert_out, index=False)
        print(f"[MAIN] Saved BERT Voita summary to {bert_out}")
        print("[MAIN] Top 10 BERT heads by accuracy:")
        print(bert_df.sort_values("accuracy", ascending=False).head(10))

    # GPT2
    if run_gpt2:
        gpt2_df = compute_gpt2_voita_counts(df, log_every=log_every)
        gpt2_df.to_csv(gpt2_out, index=False)
        print(f"[MAIN] Saved GPT2 Voita summary to {gpt2_out}")
        print("[MAIN] Top 10 GPT2 heads by accuracy:")
        print(gpt2_df.sort_values("accuracy", ascending=False).head(10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-pass Voita-style head accuracy for BERT and GPT2."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=N_SUBJ_ROOT_CSV,
        help=f"Path to nsubj-root dataset CSV (default: {N_SUBJ_ROOT_CSV}).",
    )
    parser.add_argument(
        "--bert-out",
        type=str,
        default="results/bert_voita_head_accuracy_full.csv",
        help="Output CSV for BERT head accuracies.",
    )
    parser.add_argument(
        "--gpt2-out",
        type=str,
        default="results/gpt2_voita_head_accuracy_full.csv",
        help="Output CSV for GPT2 head accuracies.",
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
        help="Maximum number of rows to process for testing purposes. Processes all rows if not specified.",
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="Skip BERT analysis (only run GPT2).",
    )
    parser.add_argument(
        "--skip-gpt2",
        action="store_true",
        help="Skip GPT2 analysis (only run BERT).",
    )

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        bert_out=args.bert_out,
        gpt2_out=args.gpt2_out,
        log_every=args.log_every,
        max_rows=args.max_rows,
        run_bert=not args.skip_bert,
        run_gpt2=not args.skip_gpt2,
    )