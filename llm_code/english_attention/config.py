# config.py

# UD treebank URLs
# Based on entries in english_attention_surprisal.ipynb
URLS = {
    "dev_gum": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/refs/heads/master/en_gum-ud-dev.conllu",
    "test_gum": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/refs/heads/master/en_gum-ud-test.conllu",
    "train_gum": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/refs/heads/master/en_gum-ud-train.conllu",
    "dev_lines": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-LinES/refs/heads/master/en_lines-ud-dev.conllu",
    "test_lines": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-LinES/refs/heads/master/en_lines-ud-test.conllu",
    "train_lines": "https://raw.githubusercontent.com/UniversalDependencies/UD_English-LinES/refs/heads/master/en_lines-ud-train.conllu",
}

# 1M Sentence Token Corpus (Leipzig)
# Using English News 2020 1M corpus (verified valid)
LEIPZIG_CORPUS_URL = "https://downloads.wortschatz-leipzig.de/corpora/eng_news_2020_1M.tar.gz"
LEIPZIG_TAR_PATH = "../../data/eng_news_2020_1M.tar.gz"
LEIPZIG_SENTENCES_PATH = "../../data/eng_news_2020_1M-sentences.txt"
MILLION_CONLLU_PATH = "../../data/eng_news_2020_1M.conllu"

# UDPipe Configuration
UDPIPE_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
UDPIPE_MODEL_NAME = "english-ewt-ud-2.15-241121"
UDPIPE_BATCH_SIZE = 50
UDPIPE_SLEEP_BETWEEN_CALLS = 0.1
JUNK_PATTERN_STR = r"\d{4}-\d{2}-\d{2}\s+tarihinde eklendi" # Keeping regex but might need adjustment for English junk if any

# Models
BERT_MODEL_NAME = "bert-base-uncased"
GPT2_MODEL_NAME = "gpt2"

# Output Paths
N_SUBJ_ROOT_CSV = "../../data/nsubj_root_sentences_en.csv"
BERT_ATTENTION_CSV = "../../data/english_bert_attention_nsubj_root.csv"
GPT2_ATTENTION_CSV = "../../data/gpt2_english_attention_nsubj_root.csv"
