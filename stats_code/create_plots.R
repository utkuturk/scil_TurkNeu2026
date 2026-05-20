#!/usr/bin/env Rscript

library(tidyverse)

project_root <- normalizePath("..")
figures_dir <- "."

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

condition_levels <- c(
  "Syncretic - Plural",
  "Syncretic - Singular",
  "Non-syncretic - Plural",
  "Non-syncretic - Singular"
)
condition_labels <- c(
  "SynPl",
  "SynSg",
  "NonSynPl",
  "NonSynSg"
)
condition_positions <- c(
  "Syncretic - Plural" = 1.00,
  "Syncretic - Singular" = 1.30,
  "Non-syncretic - Plural" = 1.60,
  "Non-syncretic - Singular" = 1.90
)

standardize_data <- function(df, lang) {
  has_model_type <- "model_type" %in% names(df)

  if (lang == "english") {
    df <- df %>%
      mutate(
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
        headn = "sg",
        attn = attractor,
        verbn = verb,
        syn = ifelse(case == "amb", 1, 0),
        lg = "german"
      )
  }

  df <- df %>%
    mutate(
      Syncretism = case_when(
        # Russian: ACC set is syn=0 (syncretic), GEN set is syn=1 (non-syncretic).
        lang == "russian" & syn == 0 ~ "Syncretic",
        lang == "russian" & syn == 1 ~ "Non-syncretic",
        # English: possessive-marked items in eng_syn_stimuli are non-syncretic.
        lang == "english" & syn == 1 ~ "Non-syncretic",
        lang == "english" & syn == 0 ~ "Syncretic",
        TRUE ~ ifelse(syn == 1, "Syncretic", "Non-syncretic")
      ),
      Attractor = ifelse(tolower(attn) == "sg", "Singular", "Plural"),
      Grammaticality = ifelse(tolower(headn) == tolower(verbn), "Grammatical", "Ungrammatical"),
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

load_data <- function(files, data_type) {
  all_data <- list()

  for (lang in names(files)) {
    path <- files[[lang]]
    if (!file.exists(path)) {
      next
    }

    df <- read_csv(path, show_col_types = FALSE)
    df <- standardize_data(df, lang)

    if (lang == "russian") {
      df <- df %>% filter(Head == "Singular")
    }

    if (data_type == "surprisal") {
      df <- df %>% filter(Model == "GPT2")
    }

    all_data[[lang]] <- df
    cat("Loaded", lang, data_type, ":", nrow(df), "rows\n")
  }

  bind_rows(all_data)
}

theme_paper <- function() {
  theme_classic(base_size = 11) +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 10),
      legend.position = "none",
      axis.title = element_text(size = 12, face = "bold"),
      axis.text.y = element_text(size = 10),
      axis.text.x = element_text(size = 12, angle = 45, hjust = 1, vjust = 1),
      panel.grid.major.y = element_line(color = "gray90", linetype = "dashed")
    )
}

build_summary <- function(data, y_col) {
  data %>%
    filter(
      !is.na(.data[[y_col]]),
      Syncretism %in% c("Syncretic", "Non-syncretic"),
      Attractor %in% c("Plural", "Singular"),
      Grammaticality %in% c("Grammatical", "Ungrammatical")
    ) %>%
    group_by(Language, Syncretism, Attractor, Grammaticality) %>%
    summarise(
      mean_val = mean(.data[[y_col]], na.rm = TRUE),
      se_val = sd(.data[[y_col]], na.rm = TRUE) / sqrt(n()),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(
      Syncretism = factor(Syncretism, levels = c("Syncretic", "Non-syncretic")),
      Condition = factor(paste(Syncretism, Attractor, sep = " - "), levels = condition_levels),
      ConditionPos = unname(condition_positions[as.character(Condition)])
    )
}

plot_by_language <- function(data, y_col, y_label, suffix) {
  summary_data <- build_summary(data, y_col)

  if (nrow(summary_data) == 0) {
    cat("No rows available for", y_col, "\n")
    return(invisible(NULL))
  }

  for (lang in sort(unique(summary_data$Language))) {
    lang_data <- summary_data %>% filter(Language == lang)
    if (nrow(lang_data) == 0) {
      next
    }

    if (lang == "Russian") {
      russian_positions <- c(
        "Non-syncretic - Singular" = 1.00,
        "Non-syncretic - Plural" = 1.30,
        "Syncretic - Singular" = 1.60,
        "Syncretic - Plural" = 1.90
      )
      lang_data <- lang_data %>%
        mutate(ConditionPos = unname(russian_positions[as.character(Condition)]))
      x_labels <- c(
        "GEN SG",
        "GEN PL",
        "ACC SG",
        "ACC PL"
      )
      x_breaks <- unname(russian_positions[c(
        "Non-syncretic - Singular",
        "Non-syncretic - Plural",
        "Syncretic - Singular",
        "Syncretic - Plural"
      )])
      x_title <- "Case/Number"
    } else {
      x_labels <- condition_labels
      x_breaks <- unname(condition_positions[condition_levels])
      x_title <- "Condition"
    }

    y_axis_label <- paste(lang, y_label, sep = "\n")

    p <- ggplot(lang_data, aes(x = ConditionPos, y = mean_val, color = Syncretism)) +
      geom_point(size = 2.8) +
      geom_errorbar(aes(ymin = mean_val - se_val, ymax = mean_val + se_val), width = 0.06, linewidth = 0.8) +
      facet_wrap(~ Grammaticality, nrow = 1, scales = "free_y") +
      scale_color_manual(
        values = c("Syncretic" = "#E69F00", "Non-syncretic" = "#0072B2"),
        labels = c("Syn", "Non-syn")
      ) +
      scale_x_continuous(
        breaks = x_breaks,
        labels = x_labels,
        limits = c(0.84, 2.06),
        expand = expansion(mult = c(0, 0))
      ) +
      labs(x = x_title, y = y_axis_label, color = NULL) +
      theme_paper()

    output_file <- file.path(figures_dir, paste0(tolower(lang), "_", suffix, ".pdf"))
    ggsave(output_file, p, width = 3.35, height = 2.7, dpi = 300)
    cat("Saved:", output_file, "\n")
  }
}

main <- function() {
  cat("Loading data...\n")
  attention_data <- load_data(attention_files, "attention")
  surprisal_data <- load_data(surprisal_files, "surprisal")

  cat("Creating per-language plots...\n")

  if ("top5_entropy" %in% names(attention_data)) {
    plot_by_language(attention_data, "top5_entropy", "Attention Entropy", "entropy")
  }

  if ("surprisal" %in% names(surprisal_data)) {
    plot_by_language(surprisal_data, "surprisal", "Surprisal", "surprisal")
  }

  cat("Done.\n")
}

main()
