#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(nlme)
})

project_root <- normalizePath("..")
out_file <- "model_results_full_interactions.csv"

attention_files <- list(
  english = file.path(project_root, "results", "stimuli_multihead", "english_multihead_stimuli_attention.csv"),
  german = file.path(project_root, "results", "stimuli_multihead", "german_multihead_stimuli_attention.csv"),
  russian = file.path(project_root, "results", "stimuli_multihead", "russian_multihead_stimuli_attention.csv"),
  turkish = file.path(project_root, "results", "stimuli_multihead", "turkish_multihead_stimuli_attention.csv")
)

surprisal_files <- list(
  english = file.path(project_root, "llm_code", "english_attention", "results", "english_surprisal_results.csv"),
  german = file.path(project_root, "llm_code", "german_attention", "results", "german_surprisal_results.csv"),
  russian = file.path(project_root, "llm_code", "russian_attention", "results", "russian_surprisal_results.csv"),
  turkish = file.path(project_root, "llm_code", "turkish_attention", "results", "turkish_surprisal_results.csv")
)

standardize_data <- function(df, lang) {
  has_model_type <- "model_type" %in% names(df)

  if (lang == "english") {
    df <- df %>%
      mutate(
        item_id = Item,
        headn = "sg",
        attn = ifelse(tolower(NP_Number) == "singular", "sg", "pl"),
        verbn = case_when(
          tolower(Auxiliary) %in% c("is", "was") ~ "sg",
          tolower(Auxiliary) %in% c("are", "were") ~ "pl",
          TRUE ~ NA_character_
        ),
        syn = ifelse(grepl("syn", source_file) & !grepl("no_syn", source_file), 1, 0),
        lg = "english"
      )
  } else if (lang == "german") {
    df <- df %>%
      mutate(
        item_id = item,
        headn = "sg",
        attn = attractor,
        verbn = verb,
        syn = ifelse(case == "amb", 1, 0),
        lg = "german"
      )
  } else {
    df <- df %>%
      mutate(
        item_id = item,
        lg = str_to_lower(lg)
      )
  }

  df <- df %>%
    mutate(
      Syncretism = ifelse(syn == 1, "Syncretic", "Non-syncretic"),
      Grammaticality = ifelse(tolower(headn) == tolower(verbn), "Grammatical", "Ungrammatical"),
      Attractor = ifelse(tolower(attn) == "sg", "Singular", "Plural"),
      Head = ifelse(tolower(headn) == "sg", "Singular", "Plural"),
      Language = str_to_title(lg)
    )

  if (has_model_type) {
    df <- df %>% mutate(Model = toupper(model_type))
  } else {
    df <- df %>% mutate(Model = "BERT")
  }

  df
}

load_all <- function(files, data_type) {
  rows <- list()
  for (lang in names(files)) {
    path <- files[[lang]]
    if (!file.exists(path)) next
    df <- read_csv(path, show_col_types = FALSE) %>%
      standardize_data(lang)
    if (lang == "russian") {
      df <- df %>% filter(Head == "Singular")
    }

    if (data_type == "surprisal") {
      df <- df %>% filter(Model == "GPT2")
    }

    rows[[lang]] <- df
  }
  bind_rows(rows)
}

extract_terms <- function(fit) {
  tt <- summary(fit)$tTable
  terms <- tibble(
    term = rownames(tt),
    estimate = as.numeric(tt[, "Value"]),
    p = as.numeric(tt[, "p-value"])
  )

  two_way <- terms %>%
    filter(str_detect(term, "Grammaticality"), str_detect(term, "Attractor"), !str_detect(term, "Syncretism")) %>%
    slice(1)

  three_way <- terms %>%
    filter(str_detect(term, "Syncretism"), str_detect(term, "Grammaticality"), str_detect(term, "Attractor")) %>%
    slice(1)

  list(two_way = two_way, three_way = three_way)
}

fit_one <- function(df_lang) {
  df_lang <- df_lang %>%
    mutate(
      Item = factor(item_id),
      Syncretism = factor(Syncretism, levels = c("Non-syncretic", "Syncretic")),
      Grammaticality = factor(Grammaticality, levels = c("Grammatical", "Ungrammatical")),
      Attractor = factor(Attractor, levels = c("Singular", "Plural"))
    ) %>%
    filter(!is.na(value), !is.na(Item), !is.na(Syncretism), !is.na(Grammaticality), !is.na(Attractor))

  ctrl <- lmeControl(msMaxIter = 200, msMaxEval = 400, returnObject = TRUE, opt = "optim")
  fit <- tryCatch(
    lme(
      fixed = value ~ Syncretism * Grammaticality * Attractor,
      random = list(Item = pdDiag(~ Syncretism * Grammaticality * Attractor)),
      data = df_lang,
      method = "REML",
      control = ctrl
    ),
    error = function(e) NULL
  )
  random_structure <- "maximal_uncorrelated"
  if (is.null(fit)) {
    fit <- lme(
      fixed = value ~ Syncretism * Grammaticality * Attractor,
      random = ~ 1 | Item,
      data = df_lang,
      method = "REML",
      control = ctrl
    )
    random_structure <- "intercept_only_fallback"
  }
  terms <- extract_terms(fit)
  tibble(
    two_way_term = terms$two_way$term,
    two_way_est = terms$two_way$estimate,
    two_way_p = terms$two_way$p,
    three_way_term = terms$three_way$term,
    three_way_est = terms$three_way$estimate,
    three_way_p = terms$three_way$p,
    random_structure = random_structure
  )
}

main <- function() {
  attention <- load_all(attention_files, "attention") %>%
    transmute(
      Language,
      item_id,
      Syncretism,
      Grammaticality,
      Attractor,
      value = top5_attn_diff,
      measure = "attn_diff"
    )

  surprisal <- load_all(surprisal_files, "surprisal") %>%
    transmute(
      Language,
      item_id,
      Syncretism,
      Grammaticality,
      Attractor,
      value = surprisal,
      measure = "surprisal"
    )

  all_data <- bind_rows(attention, surprisal)

  results <- all_data %>%
    group_by(measure, Language) %>%
    group_modify(~ fit_one(.x)) %>%
    ungroup() %>%
    arrange(measure, Language)

  write_csv(results, out_file)
  print(results, n = Inf)
  cat("Saved:", out_file, "\n")
}

main()
