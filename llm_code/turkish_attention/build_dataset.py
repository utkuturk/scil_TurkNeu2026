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
        
        # Helper to find token by ID (1-based index in conllu)
        # sent is a TokenList, indexing it is 0-based. tok['id'] is 1-based.
        # But conllu TokenList should support indexing by id or we can map it.
        # Actually in python-conllu, sent[i] gives the token at index i.
        # IDs usually match index+1. But let's be safe and build a lookup if needed, 
        # or just iterate. Actually, `sent` is a list of tokens. 
        # Token['id'] is usually an int. Head is an int.
        
        # Create a map from id -> token for easy lookup of head
        id_to_token = {tok["id"]: tok for tok in sent}

        for nsubj in nsubj_tokens:
            head_id = nsubj["head"]
            
            # Find the controller token. 
            # If head_id is 0, it points to the virtual root. 
            # Usually the token with deprel='root' is the one with head=0.
            # But the 'head' field of nsubj will point to the ID of that root token, NOT 0.
            # EXCEPT if nsubj is the root itself? No, nsubj depends on something.
            # Wait, if nsubj depends on root node, head_id is the id of the root node.
            
            if head_id == 0:
                # specific case where nsubj might be attached to root directly but 
                # practically head_id=0 usually means the token IS the root. nsubj shouldn't be root.
                # If for some reason head_id is 0, we skip or handle gracefully.
                # In standard UD, relations point to token IDs.
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
        sentences = fetch_and_parse(url)
        rows = extract_nsubj_root_rows(sentences, split_name)
        print(f"  -> {len(rows)} sentences with nsubj+root")
        all_rows.extend(rows)

    if million_path is not None:
        print(f"Parsing million-sentence corpus from {million_path} ...")
        sentences = parse_local_conllu(million_path)
        rows = extract_nsubj_root_rows(sentences, "million")
        print(f"  -> {len(rows)} sentences with nsubj+root (million)")
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def main():
    df = build_nsubj_root_df(URLS, MILLION_CONLLU_PATH)
    os.makedirs(os.path.dirname(N_SUBJ_ROOT_CSV), exist_ok=True)
    df.to_csv(N_SUBJ_ROOT_CSV, index=False)
    print(f"Saved {len(df)} sentences (including million corpus) to {N_SUBJ_ROOT_CSV}")


if __name__ == "__main__":
    main()
