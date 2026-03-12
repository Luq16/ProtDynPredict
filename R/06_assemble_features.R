#!/usr/bin/env Rscript
# ==============================================================================
# 06_assemble_features.R
# Merge all feature sets and create train/predict splits
# Input:  All feature CSVs from steps 02-05 + protein_metadata.rds
# Output: data/processed/feature_matrix_train.csv
#         data/processed/feature_matrix_predict.csv
#         data/processed/feature_summary.txt
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
})

# --- Configuration ---
CONFIG <- list(
  metadata_file   = "data/processed/protein_metadata.rds",
  seq_features    = "data/processed/sequence_features.csv",
  go_features     = "data/processed/go_features.csv",
  net_features    = "data/processed/network_features.csv",
  det_features    = "data/processed/detectability_features.csv",
  output_train    = "data/processed/feature_matrix_train.csv",
  output_predict  = "data/processed/feature_matrix_predict.csv",
  output_summary  = "data/processed/feature_summary.txt",
  # Label thresholds (0.5 FC for proteomics — 1.0 too strict for CPTAC)
  fc_threshold    = 0.5,
  pval_threshold  = 0.05,
  unchanged_pval  = 0.20,
  unchanged_fc    = 0.25
)

cat("=== Phase 1.6: Assemble Feature Matrix ===\n")

# --- Load metadata ---
metadata <- readRDS(CONFIG$metadata_file)
de_data <- metadata$de_data

# --- Load all feature sets ---
cat("Loading feature sets...\n")

seq_feat <- read_csv(CONFIG$seq_features, show_col_types = FALSE)
cat(sprintf("  Sequence features:      %d proteins x %d features\n",
            nrow(seq_feat), ncol(seq_feat) - 1))

go_feat <- read_csv(CONFIG$go_features, show_col_types = FALSE)
cat(sprintf("  GO features:            %d proteins x %d features\n",
            nrow(go_feat), ncol(go_feat) - 1))

net_feat <- read_csv(CONFIG$net_features, show_col_types = FALSE)
cat(sprintf("  Network features:       %d proteins x %d features\n",
            nrow(net_feat), ncol(net_feat) - 1))

det_feat <- read_csv(CONFIG$det_features, show_col_types = FALSE)
cat(sprintf("  Detectability features: %d proteins x %d features\n",
            nrow(det_feat), ncol(det_feat) - 1))

# --- Merge all features ---
cat("Merging feature sets...\n")

feature_matrix <- seq_feat %>%
  full_join(go_feat, by = "UniProt_ID") %>%
  full_join(net_feat, by = "UniProt_ID") %>%
  full_join(det_feat, by = "UniProt_ID")

cat(sprintf("  Merged matrix: %d proteins x %d features\n",
            nrow(feature_matrix), ncol(feature_matrix) - 1))

# --- Assign labels ---
cat("Assigning labels...\n")

labels <- de_data %>%
  dplyr::select(UniProt_ID, log2FC, adj_pvalue) %>%
  mutate(
    label = case_when(
      log2FC > CONFIG$fc_threshold & adj_pvalue < CONFIG$pval_threshold ~ "up",
      log2FC < -CONFIG$fc_threshold & adj_pvalue < CONFIG$pval_threshold ~ "down",
      abs(log2FC) < CONFIG$unchanged_fc & adj_pvalue > CONFIG$unchanged_pval ~ "unchanged",
      TRUE ~ "ambiguous"
    ),
    is_detected = TRUE
  )

feature_matrix <- feature_matrix %>%
  left_join(labels, by = "UniProt_ID") %>%
  mutate(is_detected = replace_na(is_detected, FALSE))

# --- Summary ---
label_counts <- feature_matrix %>%
  filter(is_detected) %>%
  count(label)

cat("\nLabel distribution (detected proteins):\n")
for (i in seq_len(nrow(label_counts))) {
  cat(sprintf("  %-12s %d\n", label_counts$label[i], label_counts$n[i]))
}

# --- Split into train and predict sets ---
cat("\nSplitting into train/predict sets...\n")

# Training set: detected proteins with non-ambiguous labels
train_set <- feature_matrix %>%
  filter(is_detected, label != "ambiguous") %>%
  dplyr::select(-is_detected)

# Predict set: undetected proteins (no labels)
# NOTE: This will be empty if all proteins in feature files came from DE data.
# That is expected — undetected proteins must be added separately (e.g., from
# the full human proteome or a list of proteins of interest).
# The predict set is populated when the user provides external protein IDs
# that were NOT in the DE results but have features computed.
predict_set <- feature_matrix %>%
  filter(!is_detected) %>%
  dplyr::select(-label, -log2FC, -adj_pvalue, -is_detected)

cat(sprintf("  Training set:   %d proteins (%d features + label)\n",
            nrow(train_set), ncol(train_set) - 4))  # minus UniProt_ID, label, log2FC, adj_pvalue
cat(sprintf("  Predict set:    %d proteins\n", nrow(predict_set)))
if (nrow(predict_set) == 0) {
  cat("  NOTE: Predict set is empty. This is normal if you only provided DE data.\n")
  cat("  To predict undetected proteins, add their UniProt IDs to the feature pipeline.\n")
}

# --- Handle missing values ---
cat("Handling missing values...\n")

# Count NAs per feature
na_counts_train <- train_set %>%
  dplyr::select(where(is.numeric)) %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "n_na") %>%
  filter(n_na > 0) %>%
  arrange(desc(n_na))

if (nrow(na_counts_train) > 0) {
  cat(sprintf("  Features with NAs in training set: %d\n", nrow(na_counts_train)))

  # Remove features missing in >50% of training samples
  high_na_features <- na_counts_train %>%
    filter(n_na > nrow(train_set) * 0.5) %>%
    pull(feature)

  if (length(high_na_features) > 0) {
    cat(sprintf("  Removing %d features with >50%% missing values\n",
                length(high_na_features)))
    train_set <- train_set %>% dplyr::select(-all_of(high_na_features))
    predict_set <- predict_set %>% dplyr::select(-any_of(high_na_features))
  }

  # Fill remaining NAs with column median
  numeric_cols <- train_set %>% dplyr::select(where(is.numeric)) %>% colnames()
  medians <- train_set %>%
    summarise(across(all_of(numeric_cols), ~ median(., na.rm = TRUE)))

  for (col in numeric_cols) {
    med_val <- medians[[col]]
    if (!is.na(med_val)) {
      train_set[[col]] <- replace_na(train_set[[col]], med_val)
      if (col %in% colnames(predict_set)) {
        predict_set[[col]] <- replace_na(predict_set[[col]], med_val)
      }
    }
  }
}

# --- Remove zero-variance features ---
numeric_train <- train_set %>% dplyr::select(where(is.numeric))
zero_var <- names(numeric_train)[apply(numeric_train, 2, var, na.rm = TRUE) == 0]
if (length(zero_var) > 0) {
  cat(sprintf("  Removing %d zero-variance features\n", length(zero_var)))
  train_set <- train_set %>% dplyr::select(-all_of(zero_var))
  predict_set <- predict_set %>% dplyr::select(-any_of(zero_var))
}

# --- Remove highly correlated features (|r| > 0.95) ---
cat("Removing highly correlated features...\n")
numeric_train <- train_set %>% dplyr::select(where(is.numeric)) %>% dplyr::select(-any_of(c("log2FC", "adj_pvalue")))
cor_matrix <- cor(numeric_train, use = "pairwise.complete.obs")
cor_matrix[is.na(cor_matrix)] <- 0
find_high_cor <- function(cor_mat, cutoff = 0.95) {
  cor_mat <- abs(cor_mat)
  diag(cor_mat) <- 0
  to_remove <- character()

  while (TRUE) {
    max_cor <- max(cor_mat, na.rm = TRUE)
    if (max_cor < cutoff) break

    # Find the pair
    idx <- which(cor_mat == max_cor, arr.ind = TRUE)[1, ]
    feat1 <- rownames(cor_mat)[idx[1]]
    feat2 <- colnames(cor_mat)[idx[2]]

    # Remove the one with higher mean absolute correlation
    mean1 <- mean(cor_mat[feat1, ], na.rm = TRUE)
    mean2 <- mean(cor_mat[feat2, ], na.rm = TRUE)
    remove_feat <- if (mean1 >= mean2) feat1 else feat2

    to_remove <- c(to_remove, remove_feat)
    cor_mat <- cor_mat[rownames(cor_mat) != remove_feat,
                        colnames(cor_mat) != remove_feat, drop = FALSE]
  }
  to_remove
}

high_cor_features <- find_high_cor(cor_matrix)
if (length(high_cor_features) > 0) {
  cat(sprintf("  Removing %d highly correlated features (|r| > 0.95)\n",
              length(high_cor_features)))
  train_set <- train_set %>% dplyr::select(-all_of(high_cor_features))
  predict_set <- predict_set %>% dplyr::select(-any_of(high_cor_features))
}

# --- Save ---
cat("Saving final feature matrices...\n")
dir.create(dirname(CONFIG$output_train), recursive = TRUE, showWarnings = FALSE)
write_csv(train_set, CONFIG$output_train)
write_csv(predict_set, CONFIG$output_predict)

# --- Write summary ---
n_features <- ncol(train_set) - 4  # minus UniProt_ID, label, log2FC, adj_pvalue

summary_text <- sprintf(
"ProtDynPredict Feature Matrix Summary
======================================
Date: %s

Training Set:
  Proteins:       %d
  Features:       %d
  Labels:
    Upregulated:   %d
    Downregulated: %d
    Unchanged:     %d
    Excluded:      %d (ambiguous)

Prediction Set:
  Proteins:       %d

Feature Groups:
  Sequence (protr):     ~940 -> %d after filtering
  GO similarity:        ~159
  Network (PPI):        ~15
  Detectability:        ~11

Label Thresholds:
  Upregulated:   log2FC > %.1f AND adj_p < %.2f
  Downregulated: log2FC < -%.1f AND adj_p < %.2f
  Unchanged:     |log2FC| < %.1f AND adj_p > %.2f
  Excluded:      everything else (ambiguity zone)

Files:
  Training:   %s
  Prediction: %s
",
  Sys.Date(),
  nrow(train_set), n_features,
  sum(train_set$label == "up"),
  sum(train_set$label == "down"),
  sum(train_set$label == "unchanged"),
  sum(labels$label == "ambiguous"),
  nrow(predict_set),
  n_features,
  CONFIG$fc_threshold, CONFIG$pval_threshold,
  CONFIG$fc_threshold, CONFIG$pval_threshold,
  CONFIG$unchanged_fc, CONFIG$unchanged_pval,
  CONFIG$output_train, CONFIG$output_predict
)

writeLines(summary_text, CONFIG$output_summary)

cat(summary_text)
cat("\n=== Done ===\n")
