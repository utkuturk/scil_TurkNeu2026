# scil_TurkNeu2026

Code and data for *Quantifying the cross-linguistic effects of syncretism on agreement attraction* (Turk & Neu, SCiL 2026).

We use GPT-2 surprisal and BERT attention entropy as processing proxies to ask whether morphological syncretism modulates agreement attraction the same way across English, German, Russian, and Turkish. Short answer: mostly yes for LLMs, with Russian pseudo-plurals as the notable exception.

---

## Repository layout

```
scil_TurkNeu2026/
├── llm_code/                  # Python — corpus processing + LLM extraction
│   ├── {english,german,russian,turkish}_attention/
│   │   ├── config.py          # model names, paths, output dirs
│   │   ├── corpus2conllu.py   # download Leipzig corpus → UDPipe → CoNLL-U
│   │   ├── build_dataset.py   # extract nsubj-root pairs from CoNLL-U
│   │   ├── voita_run_attention.py  # Voita probe: best layer per BERT head
│   │   ├── analyze_attention.py   # attention stats on the nsubj-root data
│   │   ├── analyze_surprisal.py   # GPT-2 surprisal on the nsubj-root data
│   │   └── utils.py           # shared: model loading, subword alignment
│   ├── voita_multihead.py     # multi-head BERT attention extraction (stimuli)
│   ├── voita_bert_leftward.py # leftward-attention variant
│   └── analyze_stimuli_multihead.py  # run attention on experimental stimuli
│
├── stats_code/                # R — Bayesian mixed-effects models + plots
│   ├── brms_main_models.R     # main 3-way BRMS models (all languages)
│   ├── brms_russian_nonsyn.R  # Russian non-syncretic-only model
│   ├── fit_interaction_models.R  # frequentist interaction fits
│   └── create_plots.R         # figures used in the paper
│
├── stimuli/                   # experimental items
│   ├── eng_syn_stimuli.csv      # English syncretic (acc plural) stimuli
│   ├── eng_no_syn_stimuli.csv   # English non-syncretic (gen plural) stimuli
│   ├── german_stimuli.csv       # German die/den stimuli (Hartsuiker et al. 2003)
│   └── tr_rus_all_conditions.csv  # Turkish & Russian items (all conditions)
│
└── data/                      # pre-computed outputs (skip heavy pipeline steps)
    ├── layer_probe/           # per-layer BERT accuracy for nsubj→root
    ├── attention_results/     # nsubj-root BERT attention by (layer, head)
    ├── surprisal_results/     # nsubj-root GPT-2 surprisal
    ├── stimuli_attention/     # stimuli-level BERT attention (input to R models)
    └── voita_head_accuracy/   # Voita per-head accuracy for layer selection
```

---

## Quick start

### Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first model run downloads ~400 MB per model (cached by HuggingFace afterwards). CUDA is used automatically if available; falls back to CPU.

### R environment

```r
install.packages(c("tidyverse", "brms", "posterior", "ggplot2", "patchwork"))
```

---

## Pipeline

### Step 1 — Corpus → CoNLL-U

Downloads 1M-sentence Leipzig corpora and parses them via the UDPipe REST API.

```bash
cd llm_code/<lang>_attention
python corpus2conllu.py
```

Output: `*.conllu` file (not included — too large).

> Corpus URLs are defined in each language's `config.py`. UDPipe API calls are rate-limited (0.1 s delay, batch size 50).

### Step 2 — Extract nsubj-root pairs

Extracts sentences that contain an nsubj→root dependency from the CoNLL-U file and saves them as a CSV.

```bash
python build_dataset.py
```

Output: `nsubj_root_sentences_<lang>.csv` (not included — 200–400 MB per language). Pre-computed layer-selection metrics are in `data/layer_probe/`.

### Step 3 — Layer selection (Voita probe)

Determines which BERT layer best tracks subject-verb dependencies by running the Voita head-accuracy probe on the nsubj-root data.

```bash
python voita_run_attention.py
```

Selected layers used in the paper: English layer 6 (63.1%), German layer 9 (48.2%), Russian layer 8 (69.3%), Turkish layer 8 (53.0%). Pre-computed head accuracy tables are in `data/voita_head_accuracy/`.

### Step 4 — Extract attention & surprisal on stimuli

Runs BERT (multi-head) and GPT-2 on the experimental stimuli.

```bash
# From the repo root:
python llm_code/analyze_stimuli_multihead.py  # BERT attention → data/stimuli_attention/
# Surprisal is extracted per-language:
python llm_code/<lang>_attention/analyze_surprisal.py
```

Pre-computed outputs are in `data/stimuli_attention/` and `data/surprisal_results/`.

### Step 5 — Bayesian models in R

Fits the Bayesian mixed-effects models reported in the paper.

```r
# Full 3-way models (all languages, both measures):
source("stats_code/brms_main_models.R")

# Russian non-syncretic-only model:
source("stats_code/brms_russian_nonsyn.R")

# Figures:
source("stats_code/create_plots.R")
```

The models use `data/stimuli_attention/` and `data/surprisal_results/` as inputs. Fitted model objects (`.rds`) are written to `stats_code/brms_fits/` (not included — regenerate locally).

---

## Models used

| Language | BERT model | GPT-2 model |
|----------|-----------|-------------|
| English  | `bert-base-uncased` | `gpt2` |
| German   | `bert-base-german-cased` | `dbmdz/german-gpt2` |
| Russian  | `deepvk/bert-base-uncased` | `ai-forever/rugpt3small_based_on_gpt2` |
| Turkish  | `dbmdz/bert-base-turkish-128k-cased` | `redrussianarmy/gpt2-turkish-cased` |

---

## Data

**Stimuli** come from prior behavioral studies:
- English: Wagers et al. (2009); Nicol et al. (2016)
- German: Hartsuiker et al. (2003)
- Russian: Slioussar (2018)
- Turkish: Lago et al. (2019); Turk & Logačev (2024)

**nsubj-root sentences** used for layer selection are drawn from the [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download) (1M-sentence news corpora, 2020). They are not redistributed here due to size; run `corpus2conllu.py` + `build_dataset.py` to regenerate.

---

## Citation

```bibtex
@inproceedings{turkneu2025syncretism,
  title     = {Quantifying the cross-linguistic effects of syncretism on agreement attraction},
  author    = {Turk, Utku and Neu, Eva},
  booktitle = {Proceedings of the Society for Computation in Linguistics (SCiL)},
  year      = {2026}
}
```

---

## License

Code: MIT. Stimuli are adapted from published experimental materials; please cite the original sources listed above.
