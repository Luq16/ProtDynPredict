#!/usr/bin/env Rscript
# Create Supplementary Table: Welch vs limma DE comparison

suppressPackageStartupMessages(library(tidyverse))

comp <- read_csv("data/ucec/raw/welch_vs_limma_comparison.csv", show_col_types = FALSE)

# --- Table A: Summary statistics ---
summary_tbl <- tibble(
  Metric = c(
    "Total proteins compared",
    "log2FC Pearson correlation",
    "log2FC Spearman correlation",
    "-log10(adj. p) Pearson correlation",
    "Overall classification concordance",
    "Proteins classified as Up (Welch / limma)",
    "Proteins classified as Down (Welch / limma)",
    "Proteins classified as Unchanged (Welch / limma)",
    "Proteins classified as Ambiguous (Welch / limma)",
    "Proteins changing DE <-> Unchanged",
    "Proteins changing direction (Up <-> Down)",
    "Proteins changing Ambiguous <-> Unchanged"
  ),
  Value = c(
    as.character(nrow(comp)),
    sprintf("%.4f", cor(comp$limma_log2FC, comp$welch_log2FC)),
    sprintf("%.4f", cor(comp$limma_log2FC, comp$welch_log2FC, method = "spearman")),
    sprintf("%.4f", cor(-log10(comp$limma_adj_p + 1e-300),
                        -log10(comp$welch_adj_p + 1e-300))),
    sprintf("%d / %d (%.1f%%)",
            sum(comp$limma_class == comp$welch_class),
            nrow(comp),
            100 * mean(comp$limma_class == comp$welch_class)),
    sprintf("%d / %d", sum(comp$welch_class == "Up"), sum(comp$limma_class == "Up")),
    sprintf("%d / %d", sum(comp$welch_class == "Down"), sum(comp$limma_class == "Down")),
    sprintf("%d / %d", sum(comp$welch_class == "Unchanged"), sum(comp$limma_class == "Unchanged")),
    sprintf("%d / %d", sum(comp$welch_class == "Ambiguous"), sum(comp$limma_class == "Ambiguous")),
    as.character(sum(
      (comp$limma_class %in% c("Up", "Down") & comp$welch_class == "Unchanged") |
      (comp$welch_class %in% c("Up", "Down") & comp$limma_class == "Unchanged")
    )),
    as.character(sum(
      (comp$limma_class == "Up" & comp$welch_class == "Down") |
      (comp$limma_class == "Down" & comp$welch_class == "Up")
    )),
    as.character(sum(
      (comp$limma_class == "Ambiguous" & comp$welch_class == "Unchanged") |
      (comp$limma_class == "Unchanged" & comp$welch_class == "Ambiguous")
    ))
  )
)

# --- Table B: Cross-tabulation ---
ct <- as.data.frame.matrix(table(
  limma = comp$limma_class,
  welch = comp$welch_class
))
ct <- ct %>%
  rownames_to_column("limma_class")

# --- Table C: Per-protein comparison (discordant only) ---
discordant <- comp %>%
  filter(limma_class != welch_class) %>%
  select(
    Gene_Symbol = gene_symbol,
    Welch_log2FC = welch_log2FC,
    Welch_adj_p = welch_adj_p,
    Welch_Class = welch_class,
    limma_log2FC,
    limma_adj_p,
    limma_Class = limma_class
  ) %>%
  arrange(Gene_Symbol)

# --- Save ---
out_dir <- "results/ucec/manuscript"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

write_csv(summary_tbl, file.path(out_dir, "Supplementary_Table_Welch_vs_limma_summary.csv"))
write_csv(ct, file.path(out_dir, "Supplementary_Table_Welch_vs_limma_crosstab.csv"))
write_csv(discordant, file.path(out_dir, "Supplementary_Table_Welch_vs_limma_discordant.csv"))

cat("=== Supplementary Table Created ===\n\n")
cat("Files:\n")
cat(sprintf("  %s/Supplementary_Table_Welch_vs_limma_summary.csv    (%d rows)\n", out_dir, nrow(summary_tbl)))
cat(sprintf("  %s/Supplementary_Table_Welch_vs_limma_crosstab.csv   (%d rows)\n", out_dir, nrow(ct)))
cat(sprintf("  %s/Supplementary_Table_Welch_vs_limma_discordant.csv (%d rows)\n", out_dir, nrow(discordant)))
cat("\nSuggested caption:\n")
cat("  Supplementary Table SX. Comparison of differential expression classifications\n")
cat("  using Welch's t-test vs. limma moderated t-test (Ritchie et al., 2015) for\n")
cat("  9,783 proteins in the CPTAC UCEC dataset. (A) Summary statistics showing\n")
cat("  concordance metrics. (B) Cross-tabulation of classifications. (C) Per-protein\n")
cat("  details for the 260 discordant proteins, all of which shifted between\n")
cat("  Ambiguous and Unchanged categories.\n")
