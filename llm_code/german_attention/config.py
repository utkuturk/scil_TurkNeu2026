# config.py

# UD treebank URLs
# UD German GSD
URLS = {
    "dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-dev.conllu",
    "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-test.conllu",
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/refs/heads/master/de_gsd-ud-train.conllu",
}

# 1M Sentence Token Corpus (Leipzig)
# Using German News 2020 1M corpus (verified valid)
LEIPZIG_CORPUS_URL = "https://downloads.wortschatz-leipzig.de/corpora/deu_news_2020_1M.tar.gz"
LEIPZIG_TAR_PATH = "../../data/deu_news_2020_1M.tar.gz"
LEIPZIG_SENTENCES_PATH = "../../data/deu_news_2020_1M-sentences.txt"
MILLION_CONLLU_PATH = "../../data/deu_news_2020_1M.conllu"

# UDPipe Configuration
UDPIPE_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
UDPIPE_MODEL_NAME = "german-gsd-ud-2.15-241121"
UDPIPE_BATCH_SIZE = 50
UDPIPE_SLEEP_BETWEEN_CALLS = 0.1
JUNK_PATTERN_STR = r"\d{4}-\d{2}-\d{2}\s+tarihinde eklendi" 

# Models
BERT_MODEL_NAME = "bert-base-german-cased"
GPT2_MODEL_NAME = "dbmdz/german-gpt2"

# Output Paths
N_SUBJ_ROOT_CSV = "../../data/nsubj_root_sentences_de.csv"
BERT_ATTENTION_CSV = "../../data/german_bert_attention_nsubj_root.csv"
GPT2_ATTENTION_CSV = "../../data/gpt2_german_attention_nsubj_root.csv"
