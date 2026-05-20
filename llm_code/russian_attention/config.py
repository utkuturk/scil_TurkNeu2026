# config.py

# UD treebank URLs
# Based on entries in russian_attention_surprisal.ipynb
URLS = {
    # Taiga
    "taiga_dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/refs/heads/master/ru_taiga-ud-dev.conllu",
    "taiga_test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/refs/heads/master/ru_taiga-ud-test.conllu",
    "taiga_train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/refs/heads/master/ru_taiga-ud-train.conllu", # The notebook had train-a, b, c, d, e splitting, but standard repo often has fused train. I'll rely on what works or use the notebook urls if explicit.
    # Actually the notebook listed:
    "taiga_train_a": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/refs/heads/master/ru_taiga-ud-train-a.conllu",
    "taiga_train_b": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/refs/heads/master/ru_taiga-ud-train-b.conllu",
    "taiga_train_c": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/refs/heads/master/ru_taiga-ud-train-c.conllu",
    "taiga_train_d": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/refs/heads/master/ru_taiga-ud-train-d.conllu",
    "taiga_train_e": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-Taiga/refs/heads/master/ru_taiga-ud-train-e.conllu",
    
    # SynTagRus
    "syntagrus_dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/refs/heads/master/ru_syntagrus-ud-dev.conllu",
    "syntagrus_test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/refs/heads/master/ru_syntagrus-ud-test.conllu",
    # SynTagRus train is usually split too due to size
    "syntagrus_train_a": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/refs/heads/master/ru_syntagrus-ud-train-a.conllu",
    "syntagrus_train_b": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/refs/heads/master/ru_syntagrus-ud-train-b.conllu",
    "syntagrus_train_c": "https://raw.githubusercontent.com/UniversalDependencies/UD_Russian-SynTagRus/refs/heads/master/ru_syntagrus-ud-train-c.conllu",
}

# 1M Sentence Token Corpus (Leipzig)
# Using Russian News 2020 which has >13M sentences. 
LEIPZIG_CORPUS_URL = "https://downloads.wortschatz-leipzig.de/corpora/rus_news_2020_1M.tar.gz"
LEIPZIG_TAR_PATH = "../../data/rus_news_2020_1M.tar.gz"
LEIPZIG_SENTENCES_PATH = "../../data/rus_news_2020_1M-sentences.txt"
MILLION_CONLLU_PATH = "../../data/rus_news_2020_1M.conllu"

# UDPipe Configuration
UDPIPE_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
UDPIPE_MODEL_NAME = "russian-syntagrus-ud-2.15-241121"
UDPIPE_BATCH_SIZE = 50
UDPIPE_SLEEP_BETWEEN_CALLS = 0.1
JUNK_PATTERN_STR = r"\d{4}-\d{2}-\d{2}\s+tarihinde eklendi" # Likely need Russian specific junk removal if any

# Models
BERT_MODEL_NAME = "deepvk/bert-base-uncased"
GPT2_MODEL_NAME = "ai-forever/rugpt3small_based_on_gpt2"

# Output Paths
N_SUBJ_ROOT_CSV = "../../data/nsubj_root_sentences_ru.csv"
BERT_ATTENTION_CSV = "../../data/russian_bert_attention_nsubj_root.csv"
GPT2_ATTENTION_CSV = "../../data/gpt2_russian_attention_nsubj_root.csv"
