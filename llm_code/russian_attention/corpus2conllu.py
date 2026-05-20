# prepare_million_corpus.py

import os
import tarfile
import time
import re
import requests

from config import (
    LEIPZIG_CORPUS_URL,
    LEIPZIG_TAR_PATH,
    LEIPZIG_SENTENCES_PATH,
    MILLION_CONLLU_PATH,
    UDPIPE_URL,
    UDPIPE_MODEL_NAME,
    UDPIPE_BATCH_SIZE,
    UDPIPE_SLEEP_BETWEEN_CALLS,
    JUNK_PATTERN_STR,
)


junk_pattern = re.compile(JUNK_PATTERN_STR)


def download_corpus():
    """Download the Leipzig tar.gz corpus if not already present."""
    os.makedirs(os.path.dirname(LEIPZIG_TAR_PATH), exist_ok=True)

    if os.path.exists(LEIPZIG_TAR_PATH):
        print(f"[info] Corpus already downloaded: {LEIPZIG_TAR_PATH}")
        return

    print(f"[info] Downloading corpus from:\n  {LEIPZIG_CORPUS_URL}")
    r = requests.get(LEIPZIG_CORPUS_URL, stream=True)
    r.raise_for_status()

    with open(LEIPZIG_TAR_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"[info] Corpus download complete: {LEIPZIG_TAR_PATH}")


def extract_sentences_file():
    """Extract the sentences text file from the Leipzig tar.gz archive."""
    os.makedirs(os.path.dirname(LEIPZIG_SENTENCES_PATH), exist_ok=True)

    if os.path.exists(LEIPZIG_SENTENCES_PATH):
        print(f"[info] Sentences file already exists: {LEIPZIG_SENTENCES_PATH}")
        return

    print(f"[info] Extracting {LEIPZIG_SENTENCES_PATH} from {LEIPZIG_TAR_PATH} ...")
    with tarfile.open(LEIPZIG_TAR_PATH, "r:gz") as tar:
        member_to_extract = None
        # The archive name may contain subdirectories; we match by suffix
        for member in tar.getmembers():
            if member.name.endswith(os.path.basename(LEIPZIG_SENTENCES_PATH)):
                member_to_extract = member
                break

        if member_to_extract is None:
            # Fallback for different naming conventions
            for member in tar.getmembers():
                if member.name.endswith("sentences.txt"):
                    member_to_extract = member
                    break
        
        if member_to_extract is None:
             raise FileNotFoundError(
                f"Could not find {LEIPZIG_SENTENCES_PATH} or 'sentences.txt' inside {LEIPZIG_TAR_PATH}"
            )

        tar.extract(member_to_extract)
        extracted_path = member_to_extract.name

        # Move to our desired path if needed
        if extracted_path != LEIPZIG_SENTENCES_PATH:
            # Ensure parent dirs exist
            os.makedirs(os.path.dirname(LEIPZIG_SENTENCES_PATH), exist_ok=True)
            os.replace(extracted_path, LEIPZIG_SENTENCES_PATH)

    print(f"[info] Extraction complete: {LEIPZIG_SENTENCES_PATH}")


def annotate_with_udpipe():
    """
    Read cleaned sentences from the Leipzig text file and annotate them via UDPipe,
    writing a single CoNLL-U file (MILLION_CONLLU_PATH).
    """
    os.makedirs(os.path.dirname(MILLION_CONLLU_PATH), exist_ok=True)

    print(f"[info] Annotating sentences from {LEIPZIG_SENTENCES_PATH} via UDPipe REST API...")
    session = requests.Session()
    total_sentences = 0
    total_calls = 0

    with open(LEIPZIG_SENTENCES_PATH, "r", encoding="utf-8", errors="replace") as inp, \
         open(MILLION_CONLLU_PATH, "w", encoding="utf-8") as out:

        batch_sentences = []

        for line_idx, line in enumerate(inp, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            text = parts[-1].strip()
            if not text:
                continue

            # remove junk prefix pattern
            text = junk_pattern.sub("", text).strip()
            if not text:
                continue

            batch_sentences.append(text)
            total_sentences += 1

            if len(batch_sentences) >= UDPIPE_BATCH_SIZE:
                total_calls = _send_batch_to_udpipe(
                    session, batch_sentences, out, total_sentences, total_calls
                )
                batch_sentences = []

        # Final leftover batch
        if batch_sentences:
            total_calls = _send_batch_to_udpipe(
                session, batch_sentences, out, total_sentences, total_calls, last_batch=True
            )

    print(f"[info] Done. Total cleaned sentences: {total_sentences}, total API calls: {total_calls}")
    print(
        f"[info] CoNLL-U saved to: {MILLION_CONLLU_PATH}, "
        f"size (bytes): {os.path.getsize(MILLION_CONLLU_PATH)}"
    )


def _send_batch_to_udpipe(session, batch_sentences, out, total_sentences, total_calls, last_batch=False):
    """Helper to send one batch to UDPipe and write the result."""
    batch_text = "\n".join(batch_sentences)

    response = session.post(
        UDPIPE_URL,
        data={
            "data": batch_text,
            "model": UDPIPE_MODEL_NAME,
            "tokenizer": "",
            "tagger": "",
            "parser": "",
        },
    )
    response.raise_for_status()

    result_json = response.json()
    if "result" not in result_json:
        where = "last batch" if last_batch else "batch"
        print(f"[warning] No result in {where}. Response keys: {list(result_json.keys())}")
        if "error" in result_json:
            print(f"[error] API Error: {result_json['error']}")
        raise RuntimeError(f"No 'result' field in UDPipe response ({where})")

    conllu_text = result_json["result"]
    out.write(conllu_text)
    if not conllu_text.endswith("\n"):
        out.write("\n")

    total_calls += 1

    if not last_batch and total_sentences % 1000 == 0:
        print(f"[info] Sentences processed so far: {total_sentences} (API calls: {total_calls})")

    if UDPIPE_SLEEP_BETWEEN_CALLS > 0:
        time.sleep(UDPIPE_SLEEP_BETWEEN_CALLS)

    return total_calls


def main():
    download_corpus()
    extract_sentences_file()
    annotate_with_udpipe()


if __name__ == "__main__":
    main()
