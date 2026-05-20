#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(tidyverse)
  library(brms)
  library(posterior)
})

project_root <- normalizePath("..")
out_csv <- "brms_interaction_probs.csv"

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
      Language = str_to_title(lg),
      Syncretism = case_when(
        # Russian: ACC set is syn=0 (syncretic), GEN set is syn=1 (non-syncretic).
        lang == "russian" & syn == 0 ~ "Syncretic",
        lang == "russian" & syn == 1 ~ "Non-syncretic",
        # English: possessive-marked items in eng_syn_stimuli are non-syncretic.
        lang == "english" & syn == 1 ~ "Non-syncretic",
        lang == "english" & syn == 0 ~ "Syncretic",
        TRUE ~ ifelse(syn == 1, "Syncretic", "Non-syncretic")
      ),
      Syncretism = factor(Syncretism, levels = c("Syncretic", "Non-syncretic")),
      Grammaticality = factor(ifelse(tolower(headn) == tolower(verbn), "Grammatical", "Ungrammatical"),
                              levels = c("Grammatical", "Ungrammatical")),
      Attractor = factor(ifelse(tolower(attn) == "sg", "Singular", "Plural"),
                         levels = c("Singular", "Plural")),
      Head = factor(ifelse(tolower(headn) == "sg", "Singular", "Plural"),
                    levels = c("Singular", "Plural")),
      Item = factor(item_id)
    )

  if (has_model_type) {
    df <- df %>% mutate(Model = toupper(model_type))
  } else {
    df <- df %>% mutate(Model = "BERT")
  }

  df
}

load_attention <- function(files) {
  rows <- list()
  for (lang in names(files)) {
    path <- files[[lang]]
    if (!file.exists(path)) next
    df <- read_csv(path, show_col_types = FALSE) %>%
      standardize_data(lang)
    if (lang == "russian") {
      df <- df %>% filter(Head == "Singular")
    }
    df <- df %>%
      transmute(
        Language, Item, Syncretism, Grammaticality, Attractor,
        value = top5_attn_diff, measure = "attn_diff"
      ) %>%
      filter(!is.na(value))
    rows[[lang]] <- df
  }
  bind_rows(rows)
}

load_surprisal <- function(files) {
  rows <- list()
  for (lang in names(files)) {
    path <- files[[lang]]
    if (!file.exists(path)) next
    df <- read_csv(path, show_col_types = FALSE) %>%
      standardize_data(lang) %>%
      filter(Model == "GPT2")
    if (lang == "russian") {
      df <- df %>% filter(Head == "Singular")
    }
    df <- df %>%
      transmute(
        Language, Item, Syncretism, Grammaticality, Attractor,
        value = surprisal, measure = "surprisal"
      ) %>%
      filter(!is.na(value))
    rows[[lang]] <- df
  }
  bind_rows(rows)
}

extract_probs <- function(fit) {
  draws <- as_draws_df(fit)
  term_two <- "b_GrammaticalityUngrammatical:AttractorPlural"
  term_three_candidates <- grep(
    "^b_Syncretism.*:GrammaticalityUngrammatical:AttractorPlural$",
    names(draws),
    value = TRUE
  )
  if (length(term_three_candidates) != 1) {
    stop("Could not uniquely identify three-way term. Found: ", paste(term_three_candidates, collapse = ", "))
  }
  term_three <- term_three_candidates[[1]]

  tibble(
    term_two = term_two,
    mean_two = mean(draws[[term_two]]),
    p_two_gt_0 = mean(draws[[term_two]] > 0),
    term_three = term_three,
    mean_three = mean(draws[[term_three]]),
    p_three_gt_0 = mean(draws[[term_three]] > 0)
  )
}

fit_one <- function(d, measure_name, lang_name) {
  priors <- c(
    prior(normal(0, 2), class = "b"),
    prior(student_t(3, 0, 2.5), class = "Intercept"),
    prior(exponential(1), class = "sd"),
    prior(exponential(1), class = "sigma")
  )

  fit <- brm(
    value ~ Syncretism * Grammaticality * Attractor +
      (1 + Grammaticality * Attractor || Item),
    data = d,
    family = gaussian(),
    prior = priors,
    chains = 4,
    cores = 4,
    iter = 1000,
    warmup = 500,
    seed = 1234,
    refresh = 0,
    control = list(adapt_delta = 0.95, max_treedepth = 12)
  )

  extract_probs(fit) %>%
    mutate(measure = measure_name, Language = lang_name, .before = 1)
}

main <- function() {
  dat <- bind_rows(
    load_attention(attention_files),
    load_surprisal(surprisal_files)
  )

  measures <- c("attn_diff", "surprisal")
  langs <- c("English", "German", "Russian", "Turkish")
  out <- list()

  for (m in measures) {
    for (lg in langs) {
      d <- dat %>% filter(measure == m, Language == lg)
      cat("Fitting:", m, lg, "n=", nrow(d), "\n")
      out[[paste(m, lg, sep = "_")]] <- fit_one(d, m, lg)
    }
  }

  results <- bind_rows(out) %>%
    select(measure, Language, mean_two, p_two_gt_0, mean_three, p_three_gt_0) %>%
    arrange(measure, Language)

  write_csv(results, out_csv)
  print(results, n = Inf)
  cat("Saved:", out_csv, "\n")
}

main()
