#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(posterior)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: brms_russian_nonsyn_only.R <measure: entropy|surprisal>")
}

measure_name <- args[[1]]
if (!measure_name %in% c("entropy", "surprisal")) {
  stop("Measure must be one of: entropy, surprisal")
}

project_root <- normalizePath("..")
fit_dir <- "brms_fits"
out_csv <- "brms_russian_nonsyn_probs.csv"

attention_file <- file.path(project_root, "results", "stimuli_multihead", "russian_multihead_stimuli_attention.csv")
surprisal_file <- file.path(project_root, "llm_code", "russian_attention", "results", "russian_surprisal_results.csv")

standardize_russian <- function(df) {
  has_model_type <- "model_type" %in% names(df)

  out <- df %>%
    mutate(
      item_id = item,
      lg = str_to_lower(lg),
      Language = str_to_title(lg),
      # Russian-specific coding: syn=0 is ACC (syncretic), syn=1 is GEN (non-syncretic).
      Syncretism = factor(ifelse(syn == 0, "Syncretic", "Non-syncretic"),
                          levels = c("Syncretic", "Non-syncretic")),
      Grammaticality = factor(ifelse(tolower(headn) == tolower(verbn), "Grammatical", "Ungrammatical"),
                              levels = c("Grammatical", "Ungrammatical")),
      Attractor = factor(ifelse(tolower(attn) == "sg", "Singular", "Plural"),
                         levels = c("Singular", "Plural")),
      Head = factor(ifelse(tolower(headn) == "sg", "Singular", "Plural"),
                    levels = c("Singular", "Plural")),
      Item = factor(item_id)
    )

  if (has_model_type) {
    out <- out %>% mutate(Model = toupper(model_type))
  } else {
    out <- out %>% mutate(Model = "BERT")
  }

  out
}

load_data <- function(measure) {
  if (measure == "entropy") {
    read_csv(attention_file, show_col_types = FALSE) %>%
      standardize_russian() %>%
      filter(Language == "Russian", Syncretism == "Non-syncretic", Head == "Singular") %>%
      transmute(Item, Grammaticality, Attractor, value = top5_entropy) %>%
      filter(!is.na(value))
  } else {
    read_csv(surprisal_file, show_col_types = FALSE) %>%
      standardize_russian() %>%
      filter(Language == "Russian", Syncretism == "Non-syncretic", Head == "Singular", Model == "GPT2") %>%
      transmute(Item, Grammaticality, Attractor, value = surprisal) %>%
      filter(!is.na(value))
  }
}

extract_probs <- function(fit) {
  draws <- as_draws_df(fit)
  term_gxa <- "b_GrammaticalityUngrammatical:AttractorPlural"
  term_a <- "b_AttractorPlural"

  if (!term_gxa %in% names(draws)) stop("Missing term: ", term_gxa)
  if (!term_a %in% names(draws)) stop("Missing term: ", term_a)

  ungram_plural_minus_singular <- draws[[term_a]] + draws[[term_gxa]]
  ungram_singular_minus_plural <- -ungram_plural_minus_singular

  tibble(
    measure = measure_name,
    mean_gxa = mean(draws[[term_gxa]]),
    p_gxa_gt_0 = mean(draws[[term_gxa]] > 0),
    mean_ungram_plural_minus_singular = mean(ungram_plural_minus_singular),
    p_ungram_plural_minus_singular_gt_0 = mean(ungram_plural_minus_singular > 0),
    mean_ungram_singular_minus_plural = mean(ungram_singular_minus_plural),
    p_ungram_singular_minus_plural_gt_0 = mean(ungram_singular_minus_plural > 0)
  )
}

main <- function() {
  dir.create(fit_dir, showWarnings = FALSE, recursive = TRUE)

  d <- load_data(measure_name)
  cat("Fitting Russian non-syncretic", measure_name, "n=", nrow(d), "\n")

  priors <- c(
    prior(normal(0, 2), class = "b"),
    prior(student_t(3, 0, 2.5), class = "Intercept"),
    prior(exponential(1), class = "sd"),
    prior(exponential(1), class = "sigma")
  )

  fit <- brm(
    value ~ Grammaticality * Attractor +
      (1 + Grammaticality * Attractor || Item),
    data = d,
    family = gaussian(),
    prior = priors,
    chains = 2,
    cores = 2,
    iter = 1000,
    warmup = 500,
    seed = 1234,
    refresh = 0,
    control = list(adapt_delta = 0.95, max_treedepth = 12)
  )

  fit_file <- file.path(fit_dir, paste0("brms_", measure_name, "_russian_nonsyn.rds"))
  saveRDS(fit, fit_file)

  row <- extract_probs(fit) %>%
    mutate(Language = "Russian", fit_file = fit_file, .before = 1)

  if (file.exists(out_csv)) {
    prev <- read_csv(out_csv, show_col_types = FALSE)
    out <- bind_rows(row, prev) %>%
      distinct(measure, Language, .keep_all = TRUE) %>%
      arrange(measure, Language)
  } else {
    out <- row
  }

  write_csv(out, out_csv)
  print(row)
  cat("Saved/updated:", out_csv, "\n")
}

main()
