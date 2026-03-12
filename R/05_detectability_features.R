#!/usr/bin/env Rscript
# ==============================================================================
# 05_detectability_features.R
# Calculate protein detectability/physicochemical features
# Input:  data/processed/protein_metadata.rds
# Output: data/processed/detectability_features.csv
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
})

# --- Configuration ---
DATASET <- Sys.getenv("DATASET", unset = "ucec")
CONFIG <- list(
  input_file  = sprintf("data/%s/processed/protein_metadata.rds", DATASET),
  output_file = sprintf("data/%s/processed/detectability_features.csv", DATASET)
)

cat("=== Phase 1.5: Detectability Features ===\n")

# --- Load metadata ---
metadata <- readRDS(CONFIG$input_file)
props <- metadata$protein_properties

cat(sprintf("  Processing %d proteins\n", nrow(props)))

# --- Calculate additional physicochemical features from sequence ---
cat("Calculating physicochemical features...\n")

calc_pI <- function(sequence) {
  if (is.na(sequence)) return(NA_real_)
  aa <- strsplit(sequence, "")[[1]]

  # pKa values for amino acid side chains
  pos_pka <- c(K = 10.5, R = 12.4, H = 6.0)
  neg_pka <- c(D = 3.65, E = 4.25, C = 8.18, Y = 10.07)
  nterm_pka <- 9.69
  cterm_pka <- 2.34

  # Count charged residues
  n_K <- sum(aa == "K"); n_R <- sum(aa == "R"); n_H <- sum(aa == "H")
  n_D <- sum(aa == "D"); n_E <- sum(aa == "E")
  n_C <- sum(aa == "C"); n_Y <- sum(aa == "Y")

  # Bisection method to find pI
  lo <- 0; hi <- 14
  for (iter in 1:100) {
    pH <- (lo + hi) / 2
    # Positive charges
    pos_charge <- n_K / (1 + 10^(pH - pos_pka["K"])) +
                  n_R / (1 + 10^(pH - pos_pka["R"])) +
                  n_H / (1 + 10^(pH - pos_pka["H"])) +
                  1 / (1 + 10^(pH - nterm_pka))
    # Negative charges
    neg_charge <- n_D / (1 + 10^(neg_pka["D"] - pH)) +
                  n_E / (1 + 10^(neg_pka["E"] - pH)) +
                  n_C / (1 + 10^(neg_pka["C"] - pH)) +
                  n_Y / (1 + 10^(neg_pka["Y"] - pH)) +
                  1 / (1 + 10^(cterm_pka - pH))

    net_charge <- sum(pos_charge) - sum(neg_charge)
    if (abs(net_charge) < 0.001) break
    if (net_charge > 0) lo <- pH else hi <- pH
  }
  pH
}

calc_charge_at_pH7 <- function(sequence) {
  if (is.na(sequence)) return(NA_real_)
  aa <- strsplit(sequence, "")[[1]]
  pH <- 7.0
  pos_pka <- c(K = 10.5, R = 12.4, H = 6.0)
  neg_pka <- c(D = 3.65, E = 4.25, C = 8.18, Y = 10.07)

  pos <- sum(aa == "K") / (1 + 10^(pH - 10.5)) +
         sum(aa == "R") / (1 + 10^(pH - 12.4)) +
         sum(aa == "H") / (1 + 10^(pH - 6.0)) +
         1 / (1 + 10^(pH - 9.69))
  neg <- sum(aa == "D") / (1 + 10^(3.65 - pH)) +
         sum(aa == "E") / (1 + 10^(4.25 - pH)) +
         sum(aa == "C") / (1 + 10^(8.18 - pH)) +
         sum(aa == "Y") / (1 + 10^(10.07 - pH)) +
         1 / (1 + 10^(2.34 - pH))
  pos - neg
}

detect_features <- props %>%
  dplyr::select(accession, sequence, seq_length, molecular_weight,
         n_tryptic_peptides, gravy_score, is_membrane) %>%
  distinct(accession, .keep_all = TRUE) %>%
  mutate(
    # Isoelectric point
    pI = map_dbl(sequence, calc_pI),

    # Net charge at pH 7
    charge_pH7 = map_dbl(sequence, calc_charge_at_pH7),

    # Fraction of aromatic residues (detectable by UV)
    frac_aromatic = map_dbl(sequence, ~ {
      if (is.na(.x)) return(NA_real_)
      aa <- strsplit(.x, "")[[1]]
      sum(aa %in% c("F", "W", "Y")) / length(aa)
    }),

    # Fraction of basic residues (affects ionization)
    frac_basic = map_dbl(sequence, ~ {
      if (is.na(.x)) return(NA_real_)
      aa <- strsplit(.x, "")[[1]]
      sum(aa %in% c("K", "R", "H")) / length(aa)
    }),

    # Fraction of hydrophobic residues
    frac_hydrophobic = map_dbl(sequence, ~ {
      if (is.na(.x)) return(NA_real_)
      aa <- strsplit(.x, "")[[1]]
      sum(aa %in% c("A", "V", "I", "L", "M", "F", "W", "P")) / length(aa)
    }),

    # Log molecular weight
    log_mw = log10(molecular_weight),

    # Tryptic peptides per 100 residues (normalized)
    tryptic_density = n_tryptic_peptides / (seq_length / 100),

    # Membrane protein (binary)
    is_membrane = as.integer(is_membrane)
  ) %>%
  dplyr::select(
    UniProt_ID = accession,
    det_seq_length = seq_length,
    det_log_mw = log_mw,
    det_pI = pI,
    det_charge_pH7 = charge_pH7,
    det_gravy = gravy_score,
    det_n_tryptic = n_tryptic_peptides,
    det_tryptic_density = tryptic_density,
    det_frac_aromatic = frac_aromatic,
    det_frac_basic = frac_basic,
    det_frac_hydrophobic = frac_hydrophobic,
    det_is_membrane = is_membrane
  )

# --- Save ---
cat("Saving detectability features...\n")
dir.create(dirname(CONFIG$output_file), recursive = TRUE, showWarnings = FALSE)
write_csv(detect_features, CONFIG$output_file)

cat(sprintf("\n=== Done ===\n"))
cat(sprintf("  Proteins: %d\n", nrow(detect_features)))
cat(sprintf("  Features: %d\n", ncol(detect_features) - 1))
cat(sprintf("  Output:   %s\n", CONFIG$output_file))
