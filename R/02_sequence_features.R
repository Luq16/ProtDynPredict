#!/usr/bin/env Rscript
# ==============================================================================
# 02_sequence_features.R
# Calculate protein sequence descriptors using protr
# Input:  data/processed/protein_metadata.rds
# Output: data/processed/sequence_features.csv
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(protr)
})

# --- Configuration ---
CONFIG <- list(
  input_file  = "data/processed/protein_metadata.rds",
  output_file = "data/processed/sequence_features.csv"
)

cat("=== Phase 1.2: Sequence Features (protr) ===\n")

# --- Load metadata ---
metadata <- readRDS(CONFIG$input_file)

# Extract sequences
sequences <- metadata$uniprot_data %>%
  dplyr::select(accession, sequence) %>%
  distinct() %>%
  filter(!is.na(sequence), nchar(sequence) >= 10)

cat(sprintf("  Processing %d protein sequences\n", nrow(sequences)))

# --- Helper: safely compute descriptor ---
safe_descriptor <- function(seq, func, name) {
  tryCatch(
    func(seq),
    error = function(e) {
      cat(sprintf("  Warning: %s failed for sequence (len=%d): %s\n",
                  name, nchar(seq), e$message))
      NULL
    }
  )
}

# --- Clean sequences (remove non-standard amino acids) ---
clean_sequence <- function(seq) {
  # protr expects only standard 20 amino acids
  cleaned <- gsub("[^ACDEFGHIKLMNPQRSTVWY]", "", toupper(seq))
  if (nchar(cleaned) < 10) return(NULL)
  cleaned
}

# --- Calculate all descriptors ---
cat("Calculating descriptors...\n")

all_features <- list()

for (i in seq_len(nrow(sequences))) {
  if (i %% 100 == 0 || i == 1) {
    cat(sprintf("  Processing protein %d/%d...\n", i, nrow(sequences)))
  }

  uid <- sequences$accession[i]
  raw_seq <- sequences$sequence[i]
  seq <- clean_sequence(raw_seq)

  if (is.null(seq)) {
    cat(sprintf("  Skipping %s: sequence too short after cleaning\n", uid))
    next
  }

  features <- list(UniProt_ID = uid)

  # 1. Amino Acid Composition (AAC) — 20 features
  aac <- safe_descriptor(seq, extractAAC, "AAC")
  if (!is.null(aac)) {
    names(aac) <- paste0("AAC_", names(aac))
    features <- c(features, as.list(aac))
  }

  # 2. Dipeptide Composition (DC) — 400 features
  dc <- safe_descriptor(seq, extractDC, "DC")
  if (!is.null(dc)) {
    names(dc) <- paste0("DC_", names(dc))
    features <- c(features, as.list(dc))
  }

  # 3. CTD Descriptors — 147 features
  # Composition (21), Transition (21), Distribution (105)
  ctd <- safe_descriptor(seq, extractCTDC, "CTDC")
  if (!is.null(ctd)) {
    names(ctd) <- paste0("CTDC_", names(ctd))
    features <- c(features, as.list(ctd))
  }

  ctdt <- safe_descriptor(seq, extractCTDT, "CTDT")
  if (!is.null(ctdt)) {
    names(ctdt) <- paste0("CTDT_", names(ctdt))
    features <- c(features, as.list(ctdt))
  }

  ctdd <- safe_descriptor(seq, extractCTDD, "CTDD")
  if (!is.null(ctdd)) {
    names(ctdd) <- paste0("CTDD_", names(ctdd))
    features <- c(features, as.list(ctdd))
  }

  # 4. Pseudo Amino Acid Composition (PseAAC) — 30 features (20 + lambda=10)
  pseaac <- safe_descriptor(seq, function(s) extractPAAC(s, lambda = 10), "PseAAC")
  if (!is.null(pseaac)) {
    names(pseaac) <- paste0("PseAAC_", seq_along(pseaac))
    features <- c(features, as.list(pseaac))
  }

  # 5. Amphiphilic PseAAC — 30 features
  apseaac <- safe_descriptor(seq, function(s) extractAPAAC(s, lambda = 10), "APseAAC")
  if (!is.null(apseaac)) {
    names(apseaac) <- paste0("APseAAC_", seq_along(apseaac))
    features <- c(features, as.list(apseaac))
  }

  # 6. Conjoint Triad Descriptors — 343 features
  ctf <- safe_descriptor(seq, extractCTriad, "CTriad")
  if (!is.null(ctf)) {
    names(ctf) <- paste0("CTriad_", names(ctf))
    features <- c(features, as.list(ctf))
  }

  # 7. Quasi-Sequence-Order Descriptors — 100 features (nlag=30)
  qso <- safe_descriptor(seq, function(s) extractQSO(s, nlag = 30), "QSO")
  if (!is.null(qso)) {
    names(qso) <- paste0("QSO_", seq_along(qso))
    features <- c(features, as.list(qso))
  }

  # 8. Sequence-Order-Coupling Number — 60 features (nlag=30)
  socn <- safe_descriptor(seq, function(s) extractSOCN(s, nlag = 30), "SOCN")
  if (!is.null(socn)) {
    names(socn) <- paste0("SOCN_", seq_along(socn))
    features <- c(features, as.list(socn))
  }

  all_features[[i]] <- features
}

# --- Combine into data frame ---
cat("Combining features into data frame...\n")

# Remove NULLs
all_features <- compact(all_features)

feature_df <- bind_rows(lapply(all_features, function(f) {
  as_tibble(f)
}))

cat(sprintf("  Final feature matrix: %d proteins x %d features\n",
            nrow(feature_df), ncol(feature_df) - 1))

# --- Remove zero-variance features ---
numeric_cols <- feature_df %>% dplyr::select(-UniProt_ID) %>% dplyr::select(where(is.numeric))
zero_var <- names(numeric_cols)[apply(numeric_cols, 2, var, na.rm = TRUE) == 0]
if (length(zero_var) > 0) {
  cat(sprintf("  Removing %d zero-variance features\n", length(zero_var)))
  feature_df <- feature_df %>% dplyr::select(-all_of(zero_var))
}

# --- Save ---
cat("Saving sequence features...\n")
dir.create(dirname(CONFIG$output_file), recursive = TRUE, showWarnings = FALSE)
write_csv(feature_df, CONFIG$output_file)

cat(sprintf("\n=== Done ===\n"))
cat(sprintf("  Proteins:  %d\n", nrow(feature_df)))
cat(sprintf("  Features:  %d\n", ncol(feature_df) - 1))
cat(sprintf("  Output:    %s\n", CONFIG$output_file))
