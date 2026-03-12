#!/usr/bin/env Rscript
# ==============================================================================
# 03_go_similarity.R
# Calculate GO semantic similarity features and GO-slim binary encoding
# Input:  data/processed/protein_metadata.rds
# Output: data/processed/go_features.csv
#
# IMPORTANT: Group-based summary features (avg similarity to up/down/unchanged)
# are computed here for the FULL dataset. During CV in Python, these must be
# recomputed per fold to avoid data leakage. This script outputs both:
#   1. GO-slim binary encoding (experiment-independent, safe for any split)
#   2. Group summary features (for reference; Python CV will recompute)
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(GOSemSim)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
  library(GO.db)
})

# --- Configuration ---
DATASET <- Sys.getenv("DATASET", unset = "ucec")
CONFIG <- list(
  input_file  = sprintf("data/%s/processed/protein_metadata.rds", DATASET),
  output_file = sprintf("data/%s/processed/go_features.csv", DATASET),
  go_slim_output = sprintf("data/%s/processed/go_slim_matrix.csv", DATASET),
  # Thresholds for labeling (must match 06_assemble_features.R)
  fc_threshold    = 0.5,
  pval_threshold  = 0.05,
  unchanged_pval  = 0.20,
  unchanged_fc    = 0.25
)

cat("=== Phase 1.3: GO Semantic Similarity Features ===\n")

# --- Load metadata ---
metadata <- readRDS(CONFIG$input_file)
de_data <- metadata$de_data
entrez_map <- metadata$entrez_map

# --- Map UniProt to Entrez ---
id_lookup <- entrez_map %>%
  dplyr::select(UNIPROT, ENTREZID) %>%
  filter(!is.na(ENTREZID)) %>%
  distinct(UNIPROT, .keep_all = TRUE)

cat(sprintf("  %d proteins with Entrez IDs\n", nrow(id_lookup)))

# --- Prepare GO data for GOSemSim ---
cat("Loading GO annotation data...\n")
go_bp_data <- godata("org.Hs.eg.db", ont = "BP")
go_mf_data <- godata("org.Hs.eg.db", ont = "MF")
go_cc_data <- godata("org.Hs.eg.db", ont = "CC")

# --- Assign labels to detected proteins ---
labeled_data <- de_data %>%
  mutate(
    label = case_when(
      log2FC > CONFIG$fc_threshold & adj_pvalue < CONFIG$pval_threshold ~ "up",
      log2FC < -CONFIG$fc_threshold & adj_pvalue < CONFIG$pval_threshold ~ "down",
      abs(log2FC) < CONFIG$unchanged_fc & adj_pvalue > CONFIG$unchanged_pval ~ "unchanged",
      TRUE ~ "ambiguous"
    )
  ) %>%
  filter(label != "ambiguous") %>%
  left_join(id_lookup, by = c("UniProt_ID" = "UNIPROT")) %>%
  filter(!is.na(ENTREZID))

cat(sprintf("  Labeled proteins: up=%d, down=%d, unchanged=%d\n",
            sum(labeled_data$label == "up"),
            sum(labeled_data$label == "down"),
            sum(labeled_data$label == "unchanged")))

# --- Part 1: GO-slim binary encoding (experiment-independent) ---
cat("Computing GO-slim binary encoding...\n")

# Get GO-slim terms (generic slim)
# Use a curated set of high-level GO terms
all_entrez <- unique(id_lookup$ENTREZID)

# Get all GO annotations for our proteins
go_annots <- tryCatch({
  AnnotationDbi::select(org.Hs.eg.db,
         keys = all_entrez,
         keytype = "ENTREZID",
         columns = "GO") %>%
    filter(!is.na(GO)) %>%
    dplyr::select(ENTREZID, GO, ONTOLOGY)
}, error = function(e) {
  cat(sprintf("  Warning: GO annotation fetch failed: %s\n", e$message))
  data.frame(ENTREZID = character(), GO = character(), ONTOLOGY = character())
})

# Find the most common GO terms to use as features (top 150)
go_term_freq <- go_annots %>%
  count(GO, ONTOLOGY, sort = TRUE) %>%
  head(150)

cat(sprintf("  Using top %d GO terms as binary features\n", nrow(go_term_freq)))

# Build binary matrix
go_slim_matrix <- matrix(0L,
                          nrow = length(all_entrez),
                          ncol = nrow(go_term_freq),
                          dimnames = list(all_entrez, go_term_freq$GO))

for (i in seq_len(nrow(go_annots))) {
  eid <- go_annots$ENTREZID[i]
  goid <- go_annots$GO[i]
  if (eid %in% rownames(go_slim_matrix) && goid %in% colnames(go_slim_matrix)) {
    go_slim_matrix[eid, goid] <- 1L
  }
}

go_slim_df <- as_tibble(go_slim_matrix, rownames = "ENTREZID") %>%
  left_join(id_lookup %>% dplyr::select(UNIPROT, ENTREZID), by = "ENTREZID") %>%
  dplyr::select(UniProt_ID = UNIPROT, everything(), -ENTREZID)

# Rename GO columns for clarity (GO:xxxx -> GOslim_N)
go_cols <- colnames(go_slim_df)[grepl("^GO:", colnames(go_slim_df))]
new_go_names <- paste0("GOslim_", seq_along(go_cols))
# Use setNames to rename directly (avoid !!! issues with special characters)
colnames(go_slim_df)[match(go_cols, colnames(go_slim_df))] <- new_go_names

# --- Part 2: Group summary features (semantic similarity to expression groups) ---
cat("Computing group semantic similarity features...\n")
cat("  (These will be recomputed per CV fold in Python to avoid leakage)\n")

set.seed(42)  # Reproducible group sampling

up_genes <- labeled_data %>% filter(label == "up") %>% pull(ENTREZID) %>% unique()
down_genes <- labeled_data %>% filter(label == "down") %>% pull(ENTREZID) %>% unique()
unch_genes <- labeled_data %>% filter(label == "unchanged") %>% pull(ENTREZID) %>% unique()

# Limit group sizes for computational feasibility
max_group_size <- 20  # Reduced from 50 for performance
if (length(up_genes) > max_group_size) up_genes <- sample(up_genes, max_group_size)
if (length(down_genes) > max_group_size) down_genes <- sample(down_genes, max_group_size)
if (length(unch_genes) > max_group_size) unch_genes <- sample(unch_genes, max_group_size)

cat(sprintf("  Reference groups: up=%d, down=%d, unchanged=%d\n",
            length(up_genes), length(down_genes), length(unch_genes)))

# Use mgeneSim for batch computation (MUCH faster than per-gene geneSim)
# Compute similarity matrix between ALL target genes and group genes in one call
target_genes <- all_entrez

# Combine all reference genes
ref_genes <- unique(c(up_genes, down_genes, unch_genes))
all_genes_for_sim <- unique(c(target_genes, ref_genes))

cat(sprintf("  Computing batch similarity for %d genes (mgeneSim)...\n",
            length(all_genes_for_sim)))

# Compute one similarity matrix per ontology using mgeneSim
compute_batch_group_sims <- function(sem_data, ont_name) {
  cat(sprintf("    %s ontology...\n", ont_name))

  sim_mat <- tryCatch({
    mgeneSim(all_genes_for_sim, semData = sem_data,
             measure = "Wang", combine = "BMA", verbose = FALSE)
  }, error = function(e) {
    cat(sprintf("    Warning: mgeneSim failed for %s: %s\n", ont_name, e$message))
    # Return empty matrix
    mat <- matrix(NA_real_, nrow = length(all_genes_for_sim),
                  ncol = length(all_genes_for_sim))
    rownames(mat) <- colnames(mat) <- all_genes_for_sim
    mat
  })

  # Extract mean similarity to each group for each target gene
  result <- tibble(ENTREZID = target_genes)

  for (group_name in c("up", "down", "unch")) {
    group_ids <- switch(group_name,
      "up" = up_genes, "down" = down_genes, "unch" = unch_genes)

    col_name <- paste0("GO_", ont_name, "_sim_", group_name)

    # For each target gene, compute mean similarity to group
    result[[col_name]] <- sapply(target_genes, function(gid) {
      grp <- setdiff(group_ids, gid)
      if (length(grp) == 0) return(NA_real_)
      if (!(gid %in% rownames(sim_mat))) return(NA_real_)
      valid_grp <- grp[grp %in% colnames(sim_mat)]
      if (length(valid_grp) == 0) return(NA_real_)
      mean(sim_mat[gid, valid_grp], na.rm = TRUE)
    })
  }

  result
}

# Compute for each ontology
bp_sims <- compute_batch_group_sims(go_bp_data, "BP")
mf_sims <- compute_batch_group_sims(go_mf_data, "MF")
cc_sims <- compute_batch_group_sims(go_cc_data, "CC")

# Combine
group_features <- bp_sims %>%
  left_join(mf_sims, by = "ENTREZID") %>%
  left_join(cc_sims, by = "ENTREZID")

# Map back to UniProt IDs
group_features <- group_features %>%
  left_join(id_lookup %>% dplyr::select(UNIPROT, ENTREZID), by = "ENTREZID") %>%
  dplyr::select(UniProt_ID = UNIPROT, starts_with("GO_")) %>%
  filter(!is.na(UniProt_ID)) %>%
  distinct(UniProt_ID, .keep_all = TRUE)

# --- Combine GO-slim and group features ---
cat("Combining GO features...\n")

go_combined <- go_slim_df %>%
  left_join(group_features, by = "UniProt_ID")

# --- Save ---
cat("Saving GO features...\n")
dir.create(dirname(CONFIG$output_file), recursive = TRUE, showWarnings = FALSE)
write_csv(go_combined, CONFIG$output_file)
write_csv(go_slim_df, CONFIG$go_slim_output)

cat(sprintf("\n=== Done ===\n"))
cat(sprintf("  Proteins:          %d\n", nrow(go_combined)))
cat(sprintf("  GO-slim features:  %d\n", ncol(go_slim_df) - 1))
cat(sprintf("  Group features:    9 (3 ontologies x 3 groups)\n"))
cat(sprintf("  Total GO features: %d\n", ncol(go_combined) - 1))
cat(sprintf("  Output: %s\n", CONFIG$output_file))
