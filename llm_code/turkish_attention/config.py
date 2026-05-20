# config.py

# UD treebank URLs
URLS = {
    "dev": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/refs/heads/master/tr_boun-ud-dev.conllu",
    "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/refs/heads/master/tr_boun-ud-test.conllu",
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-BOUN/refs/heads/master/tr_boun-ud-train.conllu",
    "dev2": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Atis/refs/heads/master/tr_atis-ud-dev.conllu",
    "test2": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Atis/refs/heads/master/tr_atis-ud-test.conllu",
    "train2": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Atis/refs/heads/master/tr_atis-ud-train.conllu",
    "test3": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-GB/refs/heads/master/tr_gb-ud-test.conllu",
    "dev4": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/refs/heads/master/tr_kenet-ud-dev.conllu",
    "test4": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/refs/heads/master/tr_kenet-ud-test.conllu",
    "train4": "https://raw.githubusercontent.com/UniversalDependencies/UD_Turkish-Kenet/refs/heads/master/tr_kenet-ud-train.conllu",
}

MILLION_CONLLU_PATH = "../../data/tur_tr_web_2015_1M.conllu"  

BERT_MODEL_NAME = "dbmdz/bert-base-turkish-128k-cased"
GPT2_MODEL_NAME = "redrussianarmy/gpt2-turkish-cased"

N_SUBJ_ROOT_CSV = "../../data/nsubj_root_sentences_tr.csv"
BERT_ATTENTION_CSV = "../../data/turkish_bert_attention_nsubj_root.csv"
GPT2_ATTENTION_CSV = "../../data/gpt2_turkish_attention_nsubj_root.csv"


LEIPZIG_CORPUS_URL = "https://downloads.wortschatz-leipzig.de/corpora/tur-tr_web_2015_1M.tar.gz"
LEIPZIG_TAR_PATH = "../../data/tur-tr_web_2015_1M.tar.gz"
LEIPZIG_SENTENCES_PATH = "../../data/tur-tr_web_2015_1M-sentences.txt"

MILLION_CONLLU_PATH = "../../data/tur_tr_web_2015_1M.conllu"

UDPIPE_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
UDPIPE_MODEL_NAME = "turkish-boun-ud-2.15-241121"
UDPIPE_BATCH_SIZE = 50
UDPIPE_SLEEP_BETWEEN_CALLS = 0.1

JUNK_PATTERN_STR = r"\d{4}-\d{2}-\d{2}\s+tarihinde eklendi"