# build_dataset.py

import os
import requests
from conllu import parse
import pandas as pd

from config import URLS, N_SUBJ_ROOT_CSV, MILLION_CONLLU_PATH


def fetch_and_parse(url: str):
    """Download a UD .conllu file and parse it into sentence objects."""
    resp = requests.get(url)
    resp.raise_for_status()
    # UD files are sometimes large; ensure we handle text encoding
    return parse(resp.text)


def parse_local_conllu(path: str):
    """Parse a local .conllu file into sentence objects."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return parse(text)


def extract_nsubj_root_rows(sentences, split_name: str):
    """
    From a sequence of parsed sentences, extract all nsubj relations.
    For each nsubj token, find its head (controller) and create a row.
    Return a list of dict rows.
    """
    rows = []

    for sent in sentences:
        # We want to process every nsubj in the sentence
        nsubj_tokens = [tok for tok in sent if tok["deprel"] == "nsubj"]
        
        if not nsubj_tokens:
            continue

        sentence_text = " ".join(tok["form"] for tok in sent)
        
        # Create a map from id -> token for easy lookup of head
        id_to_token = {tok["id"]: tok for tok in sent}

        for nsubj in nsubj_tokens:
            head_id = nsubj["head"]
            
            if head_id == 0:
                continue
            
            controller = id_to_token.get(head_id)
            if not controller:
                continue

            rows.append(
                {
                    "sentence": sentence_text,
                    "nsubj": nsubj["form"],
                    "root": controller["form"], # using 'root' key for controller as requested
                    "nsubj_first": nsubj["form"],
                    "nsubj_position": nsubj["id"],
                    "root_first": controller["form"],
                    "split": split_name,
                }
            )
            
    return rows

def build_nsubj_root_df(urls: dict, million_path: str) -> pd.DataFrame:
    """
    Build a single DataFrame that includes:
      - all UD splits (from remote URLs)
      - the local 1M-sentence web corpus
    """
    all_rows = []

    for split_name, url in urls.items():
        print(f"Fetching {split_name} from {url} ...")
        try:
            sentences = fetch_and_parse(url)
            rows = extract_nsubj_root_rows(sentences, split_name)
            print(f"  -> {len(rows)} sentences with nsubj+root")
            all_rows.extend(rows)
        except Exception as e:
            print(f"Failed to fetch/parse {split_name}: {e}")

    if million_path is not None:
        if os.path.exists(million_path):
            print(f"Parsing million-sentence corpus from {million_path} ...")
            sentences = parse_local_conllu(million_path)
            rows = extract_nsubj_root_rows(sentences, "million")
            print(f"  -> {len(rows)} sentences with nsubj+root (million)")
            all_rows.extend(rows)
        else:
            print(f"Million corpus not found at {million_path}, skipping.")

    return pd.DataFrame(all_rows)


def main():
    df = build_nsubj_root_df(URLS, MILLION_CONLLU_PATH)
    os.makedirs(os.path.dirname(N_SUBJ_ROOT_CSV), exist_ok=True)
    df.to_csv(N_SUBJ_ROOT_CSV, index=False)
    print(f"Saved {len(df)} sentences (including million corpus) to {N_SUBJ_ROOT_CSV}")


if __name__ == "__main__":
    main()
