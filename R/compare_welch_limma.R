#!/usr/bin/env Rscript
# Compare Welch's t-test vs limma DE results for UCEC

suppressPackageStartupMessages({
  library(limma)
  library(tidyverse)
})

cancer <- "ucec"
cat("=== Welch vs limma DE Comparison ===\n\n")

# --- Load expression matrix ---
expr_df <- read.csv(sprintf("data/%s/raw/expr_matrix.csv", cancer),
                    row.names = 1, check.names = FALSE)
labels  <- read.csv(sprintf("data/%s/raw/sample_labels.csv", cancer),
                    row.names = 1)

expr_mat <- t(as.matrix(expr_df))  # proteins x samples
group <- factor(labels$group[match(colnames(expr_mat), rownames(labels))],
                levels = c("Normal", "Tumor"))

cat(sprintf("  Expression matrix: %d proteins x %d samples\n", nrow(expr_mat), ncol(expr_mat)))
cat(sprintf("  Tumor: %d, Normal: %d\n\n", sum(group == "Tumor"), sum(group == "Normal")))

# --- Run limma ---
cat("Running limma...\n")
design <- model.matrix(~ 0 + group)
colnames(design) <- c("Normal", "Tumor")
contrast <- makeContrasts(Tumor - Normal, levels = design)

fit <- lmFit(expr_mat, design)
fit2 <- contrasts.fit(fit, contrast)
fit3 <- eBayes(fit2)

limma_res <- topTable(fit3, coef = 1, number = Inf, sort.by = "none") %>%
  as_tibble(rownames = "gene_symbol") %>%
  select(gene_symbol, limma_log2FC = logFC, limma_adj_p = adj.P.Val)

cat(sprintf("  limma results: %d proteins\n", nrow(limma_res)))

# --- Load existing Welch results ---
cat("Loading Welch t-test results...\n")
welch_res <- read_csv(sprintf("data/%s/raw/de_results.csv", cancer),
                      show_col_types = FALSE)

# Map UniProt back to gene symbols for comparison
# We need the gene-to-UniProt mapping used by fetch_cptac_data.py
# Instead, run Welch directly on the same expression matrix for fair comparison
cat("Running Welch t-test on same matrix for fair comparison...\n")

tumor_idx <- which(group == "Tumor")
normal_idx <- which(group == "Normal")

welch_results <- tibble(
  gene_symbol = rownames(expr_mat),
  welch_log2FC = NA_real_,
  welch_pval = NA_real_
)

for (i in seq_len(nrow(expr_mat))) {
  tumor_vals <- expr_mat[i, tumor_idx]
  normal_vals <- expr_mat[i, normal_idx]

  tumor_vals <- tumor_vals[!is.na(tumor_vals)]
  normal_vals <- normal_vals[!is.na(normal_vals)]

  if (length(tumor_vals) < 3 || length(normal_vals) < 3) next

  welch_results$welch_log2FC[i] <- mean(tumor_vals) - mean(normal_vals)
  tt <- t.test(tumor_vals, normal_vals, var.equal = FALSE)
  welch_results$welch_pval[i] <- tt$p.value
}

welch_results <- welch_results %>%
  filter(!is.na(welch_pval))

# BH correction
welch_results$welch_adj_p <- p.adjust(welch_results$welch_pval, method = "BH")

cat(sprintf("  Welch results: %d proteins\n\n", nrow(welch_results)))

# --- Merge and compare ---
merged <- inner_join(limma_res, welch_results, by = "gene_symbol")
cat(sprintf("  Proteins in both: %d\n\n", nrow(merged)))

# Label each protein under both methods
FC_THRESH <- 0.5
P_THRESH  <- 0.05
UNCH_FC   <- 0.25
UNCH_P    <- 0.20

label_de <- function(log2fc, adj_p, fc_thresh = FC_THRESH, p_thresh = P_THRESH,
                     unch_fc = UNCH_FC, unch_p = UNCH_P) {
  case_when(
    abs(log2fc) > fc_thresh & adj_p < p_thresh & log2fc > 0 ~ "Up",
    abs(log2fc) > fc_thresh & adj_p < p_thresh & log2fc < 0 ~ "Down",
    abs(log2fc) < unch_fc & adj_p > unch_p ~ "Unchanged",
    TRUE ~ "Ambiguous"
  )
}

merged <- merged %>%
  mutate(
    limma_class  = label_de(limma_log2FC, limma_adj_p),
    welch_class  = label_de(welch_log2FC, welch_adj_p)
  )

# --- Results ---
cat("=== CLASS DISTRIBUTION ===\n\n")
cat("limma:\n")
print(table(merged$limma_class))
cat("\nWelch:\n")
print(table(merged$welch_class))

cat("\n=== CROSS-TABULATION (limma rows x Welch cols) ===\n\n")
ct <- table(limma = merged$limma_class, welch = merged$welch_class)
print(ct)

# Agreement
agreed <- sum(merged$limma_class == merged$welch_class)
total  <- nrow(merged)
cat(sprintf("\n=== AGREEMENT ===\n"))
cat(sprintf("  Total proteins:     %d\n", total))
cat(sprintf("  Same class:         %d (%.1f%%)\n", agreed, 100 * agreed / total))
cat(sprintf("  Different class:    %d (%.1f%%)\n", total - agreed, 100 * (total - agreed) / total))

# Focus on non-ambiguous
non_ambig <- merged %>% filter(limma_class != "Ambiguous" | welch_class != "Ambiguous")
agreed_na <- sum(non_ambig$limma_class == non_ambig$welch_class)
cat(sprintf("\n  Non-ambiguous proteins: %d\n", nrow(non_ambig)))
cat(sprintf("  Same class (non-ambig): %d (%.1f%%)\n", agreed_na, 100 * agreed_na / nrow(non_ambig)))

# Correlation of log2FC and p-values
cat(sprintf("\n=== CORRELATION ===\n"))
cat(sprintf("  log2FC Pearson r:    %.4f\n", cor(merged$limma_log2FC, merged$welch_log2FC)))
cat(sprintf("  log2FC Spearman rho: %.4f\n", cor(merged$limma_log2FC, merged$welch_log2FC, method = "spearman")))
cat(sprintf("  -log10(p) Pearson r: %.4f\n",
            cor(-log10(merged$limma_adj_p + 1e-300), -log10(merged$welch_adj_p + 1e-300))))

# Proteins that flip between DE and Unchanged
flips <- merged %>%
  filter(
    (limma_class %in% c("Up", "Down") & welch_class == "Unchanged") |
    (welch_class %in% c("Up", "Down") & limma_class == "Unchanged")
  )
cat(sprintf("\n=== CRITICAL FLIPS (DE <-> Unchanged) ===\n"))
cat(sprintf("  Proteins flipping between DE and Unchanged: %d (%.1f%%)\n",
            nrow(flips), 100 * nrow(flips) / total))

# Direction flips (Up <-> Down)
dir_flips <- merged %>%
  filter(
    (limma_class == "Up" & welch_class == "Down") |
    (limma_class == "Down" & welch_class == "Up")
  )
cat(sprintf("  Proteins flipping direction (Up <-> Down): %d\n", nrow(dir_flips)))

# Save
out_path <- sprintf("data/%s/raw/de_results_limma.csv", cancer)
limma_out <- limma_res %>%
  select(gene_symbol, log2FC = limma_log2FC, adj_pvalue = limma_adj_p)
write_csv(limma_out, out_path)
cat(sprintf("\n  Saved limma results: %s\n", out_path))

write_csv(merged, sprintf("data/%s/raw/welch_vs_limma_comparison.csv", cancer))
cat(sprintf("  Saved comparison:   data/%s/raw/welch_vs_limma_comparison.csv\n", cancer))

cat("\n=== Done ===\n")
