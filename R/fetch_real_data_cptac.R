#!/usr/bin/env Rscript
# ==============================================================================
# fetch_real_data_cptac.R
# Download REAL tumor vs normal proteomics from CPTAC via Python cptac package
#
# Dataset: CPTAC Endometrial Cancer (or Colon Cancer)
#          Real MS-based proteomics, tumor vs normal adjacent tissue
#          ~6,000-10,000 human proteins
#
# Requires: Python with cptac package (pip install cptac)
# Runs DE analysis with limma
# Output: data/raw/de_results.csv (UniProt_ID, log2FC, adj_pvalue)
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(reticulate)
  library(limma)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
})

# --- Configuration ---
CANCER_TYPE <- "Endometrial"  # Options: Endometrial, Colon, Ovarian, Brca, RenalCcrcc

cat("=== Fetching Real CPTAC Proteomics Data ===\n")
cat(sprintf("  Cancer type: %s\n", CANCER_TYPE))
cat("  Comparison: Tumor vs Normal adjacent tissue\n\n")

# --- Check Python + cptac ---
cat("Checking Python environment...\n")
if (!py_available()) {
  stop("Python not available. Install Python and configure reticulate.\n",
       "  Options:\n",
       "  1. reticulate::install_python()\n",
       "  2. use_python('/path/to/python3')\n")
}

cptac_available <- py_module_available("cptac")
if (!cptac_available) {
  cat("  Installing cptac Python package...\n")
  py_install("cptac", pip = TRUE)
}

cptac <- import("cptac")
cat("  cptac package loaded.\n")

# --- Download CPTAC dataset ---
cat(sprintf("Downloading CPTAC %s data (first run downloads ~100MB)...\n", CANCER_TYPE))

dataset <- switch(CANCER_TYPE,
  "Endometrial" = cptac$Ucec(),
  "Colon"       = cptac$Colon(),
  "Ovarian"     = cptac$Ovarian(),
  "Brca"        = cptac$Brca(),
  "RenalCcrcc"  = cptac$Ccrcc(),
  stop("Unknown cancer type: ", CANCER_TYPE)
)

# --- Extract proteomics data ---
cat("Extracting proteomics quantification...\n")
prot_py <- dataset$get_proteomics()

# Convert to R data frame
prot_df <- py_to_r(prot_py)
cat(sprintf("  Proteomics matrix: %d samples x %d proteins\n",
            nrow(prot_df), ncol(prot_df)))

# --- Get sample metadata ---
cat("Extracting sample metadata...\n")
clinical_py <- dataset$get_clinical()
clinical_df <- py_to_r(clinical_py)

# Identify tumor vs normal samples
# CPTAC uses "Tumor" and "Normal" in Sample_Tumor_Normal column
if ("Sample_Tumor_Normal" %in% colnames(clinical_df)) {
  sample_type <- clinical_df$Sample_Tumor_Normal
} else {
  # Try to infer from sample IDs (normals often have ".N" suffix)
  sample_ids <- rownames(prot_df)
  sample_type <- ifelse(grepl("\\.N$|Normal", sample_ids), "Normal", "Tumor")
}

names(sample_type) <- rownames(clinical_df)

# Match to proteomics samples
common_samples <- intersect(rownames(prot_df), names(sample_type))
prot_df <- prot_df[common_samples, ]
sample_type <- sample_type[common_samples]

n_tumor <- sum(sample_type == "Tumor")
n_normal <- sum(sample_type == "Normal")
cat(sprintf("  Tumor samples:  %d\n", n_tumor))
cat(sprintf("  Normal samples: %d\n", n_normal))

if (n_normal < 2) {
  stop("Too few normal samples (", n_normal, "). Try a different cancer type.\n",
       "  Recommended: Endometrial or Colon (both have paired normals)")
}

# --- Filter proteins ---
cat("Filtering proteins...\n")

# Remove proteins with >50% missing values
prot_mat <- as.matrix(prot_df)
missing_frac <- colMeans(is.na(prot_mat))
keep_proteins <- missing_frac < 0.5
prot_mat <- prot_mat[, keep_proteins]
cat(sprintf("  After missing value filter: %d proteins\n", ncol(prot_mat)))

# Impute remaining NAs with column minimum (simple MNAR imputation)
for (j in seq_len(ncol(prot_mat))) {
  na_idx <- is.na(prot_mat[, j])
  if (any(na_idx)) {
    col_min <- min(prot_mat[, j], na.rm = TRUE)
    prot_mat[na_idx, j] <- col_min - abs(rnorm(sum(na_idx), mean = 0, sd = 0.3))
  }
}

# --- DE analysis with limma ---
cat("Running DE analysis (limma)...\n")

# Transpose: proteins as rows, samples as columns
expr_mat <- t(prot_mat)

# Design matrix
group <- factor(sample_type[colnames(expr_mat)], levels = c("Normal", "Tumor"))
design <- model.matrix(~ 0 + group)
colnames(design) <- c("Normal", "Tumor")

contrast_matrix <- makeContrasts(Tumor - Normal, levels = design)

fit <- lmFit(expr_mat, design)
fit2 <- contrasts.fit(fit, contrast_matrix)
fit3 <- eBayes(fit2)

results <- topTable(fit3, coef = 1, number = Inf, sort.by = "none")
cat(sprintf("  DE results: %d proteins\n", nrow(results)))

# --- Map gene names to UniProt IDs ---
cat("Mapping gene symbols to UniProt IDs...\n")

# CPTAC uses gene symbols as protein identifiers
gene_symbols <- rownames(results)

# Clean up gene symbols (CPTAC sometimes has multi-mapped genes)
gene_symbols_clean <- gsub("_.*$", "", gene_symbols)  # remove suffixes

# Map to UniProt
uniprot_map <- tryCatch({
  select(org.Hs.eg.db,
         keys = unique(gene_symbols_clean),
         keytype = "SYMBOL",
         columns = c("UNIPROT", "SYMBOL")) %>%
    distinct(SYMBOL, .keep_all = TRUE)  # one UniProt per gene
}, error = function(e) {
  cat(sprintf("  Warning: UniProt mapping failed: %s\n", e$message))
  data.frame(SYMBOL = character(), UNIPROT = character())
})

cat(sprintf("  Mapped %d/%d genes to UniProt\n",
            sum(gene_symbols_clean %in% uniprot_map$SYMBOL),
            length(unique(gene_symbols_clean))))

# --- Format output ---
results_df <- results %>%
  as_tibble(rownames = "gene") %>%
  mutate(gene_clean = gsub("_.*$", "", gene)) %>%
  left_join(uniprot_map, by = c("gene_clean" = "SYMBOL")) %>%
  filter(!is.na(UNIPROT)) %>%
  select(
    UniProt_ID = UNIPROT,
    log2FC = logFC,
    adj_pvalue = adj.P.Val
  ) %>%
  group_by(UniProt_ID) %>%
  slice_min(adj_pvalue, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  filter(!is.na(log2FC), !is.na(adj_pvalue))

# --- Summary ---
cat(sprintf("\nDE Summary (%s, Tumor vs Normal):\n", CANCER_TYPE))
cat(sprintf("  Total proteins:    %d\n", nrow(results_df)))
cat(sprintf("  Upregulated:       %d (log2FC > 1, adj_p < 0.05)\n",
            sum(results_df$log2FC > 1 & results_df$adj_pvalue < 0.05)))
cat(sprintf("  Downregulated:     %d (log2FC < -1, adj_p < 0.05)\n",
            sum(results_df$log2FC < -1 & results_df$adj_pvalue < 0.05)))
cat(sprintf("  Unchanged:         %d (|log2FC| < 0.5, adj_p > 0.20)\n",
            sum(abs(results_df$log2FC) < 0.5 & results_df$adj_pvalue > 0.20)))
cat(sprintf("  log2FC range:      [%.2f, %.2f]\n", min(results_df$log2FC), max(results_df$log2FC)))

# --- Save ---
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
write_csv(results_df, "data/raw/de_results.csv")

cat(sprintf("\n  Output: data/raw/de_results.csv\n"))
cat("  Next: Rscript R/00_run_pipeline.R\n")
cat("\n=== Done ===\n")
