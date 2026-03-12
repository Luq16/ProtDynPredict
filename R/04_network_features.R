#!/usr/bin/env Rscript
# ==============================================================================
# 04_network_features.R
# Calculate PPI network neighborhood features
# Input:  data/processed/protein_metadata.rds
# Output: data/processed/network_features.csv
#
# IMPORTANT: Neighbor expression statistics are label-dependent.
# During CV in Python, these MUST be recomputed per fold.
# This script computes them on the full dataset for reference/exploration.
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(igraph)
  library(clusterProfiler)
  library(ReactomePA)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
})

# --- Configuration ---
CONFIG <- list(
  input_file  = "data/processed/protein_metadata.rds",
  output_file = "data/processed/network_features.csv",
  pathway_output = "data/processed/pathway_membership.csv",
  # Label thresholds (must match 03 and 06)
  fc_threshold    = 1.0,
  pval_threshold  = 0.05,
  unchanged_pval  = 0.20,
  unchanged_fc    = 0.5
)

cat("=== Phase 1.4: Network Features ===\n")

# --- Load metadata ---
metadata <- readRDS(CONFIG$input_file)
de_data <- metadata$de_data
ppi <- metadata$ppi_interactions
string_map <- metadata$string_mapping
entrez_map <- metadata$entrez_map

# --- Assign labels ---
labeled_data <- de_data %>%
  mutate(
    label = case_when(
      log2FC > CONFIG$fc_threshold & adj_pvalue < CONFIG$pval_threshold ~ "up",
      log2FC < -CONFIG$fc_threshold & adj_pvalue < CONFIG$pval_threshold ~ "down",
      abs(log2FC) < CONFIG$unchanged_fc & adj_pvalue > CONFIG$unchanged_pval ~ "unchanged",
      TRUE ~ "ambiguous"
    )
  )

# --- Build PPI graph ---
cat("Building PPI network...\n")

if (nrow(ppi) == 0) {
  cat("  WARNING: No PPI interactions available. Generating empty network features.\n")

  empty_features <- tibble(
    UniProt_ID = unique(de_data$UniProt_ID),
    ppi_degree = 0L, ppi_betweenness = 0, ppi_clustering = 0,
    ppi_n_detected_neighbors = 0L,
    ppi_frac_neighbors_up = NA_real_, ppi_frac_neighbors_down = NA_real_,
    ppi_frac_neighbors_unch = NA_real_,
    ppi_weighted_frac_up = NA_real_, ppi_weighted_frac_down = NA_real_,
    ppi_weighted_frac_unch = NA_real_
  )

  write_csv(empty_features, CONFIG$output_file)
  cat("  Output: empty network features written\n")
  cat("  Skipping remaining network computations.\n")
  # Write empty pathway file
  write_csv(tibble(UniProt_ID = character(), pathway_id = character(),
                    source = character()), CONFIG$pathway_output)
} else {
# PPI data available — compute all network features

# Create STRING ID to UniProt mapping
string_to_uniprot <- string_map %>%
  dplyr::select(UniProt_ID, STRING_id) %>%
  distinct()

# Build igraph object
g <- graph_from_data_frame(
  ppi %>% dplyr::select(from, to),
  directed = FALSE,
  vertices = unique(c(ppi$from, ppi$to))
)

# Add edge weights (STRING combined score, normalized to 0-1)
E(g)$weight <- ppi$combined_score / 1000

cat(sprintf("  PPI network: %d nodes, %d edges\n", vcount(g), ecount(g)))

# --- Compute network topology features ---
cat("Computing network topology features...\n")

topo_features <- tibble(
  STRING_id = V(g)$name,
  ppi_degree = degree(g),
  ppi_betweenness = betweenness(g, normalized = TRUE),
  ppi_clustering = transitivity(g, type = "local")
) %>%
  mutate(ppi_clustering = replace_na(ppi_clustering, 0))

# Map to UniProt IDs
topo_features <- topo_features %>%
  left_join(string_to_uniprot, by = "STRING_id") %>%
  filter(!is.na(UniProt_ID)) %>%
  dplyr::select(-STRING_id)

cat(sprintf("  Topology features for %d proteins\n", nrow(topo_features)))

# --- Compute neighbor expression statistics ---
cat("Computing neighbor expression statistics...\n")
cat("  (Will be recomputed per CV fold in Python)\n")

# Map labels to STRING IDs
label_map <- labeled_data %>%
  dplyr::select(UniProt_ID, label) %>%
  left_join(string_to_uniprot, by = "UniProt_ID") %>%
  filter(!is.na(STRING_id))

# For each node in the PPI network, compute expression stats of neighbors
all_nodes <- V(g)$name
neighbor_features <- tibble(
  STRING_id = character(),
  ppi_n_detected_neighbors = integer(),
  ppi_frac_neighbors_up = double(),
  ppi_frac_neighbors_down = double(),
  ppi_frac_neighbors_unch = double(),
  ppi_weighted_frac_up = double(),
  ppi_weighted_frac_down = double(),
  ppi_weighted_frac_unch = double()
)

for (node in all_nodes) {
  # Get neighbors
  nbrs <- neighbors(g, node)$name

  # Get labeled neighbors
  nbr_labels <- label_map %>%
    filter(STRING_id %in% nbrs)

  n_detected <- nrow(nbr_labels)

  if (n_detected == 0) {
    neighbor_features <- bind_rows(neighbor_features, tibble(
      STRING_id = node,
      ppi_n_detected_neighbors = 0L,
      ppi_frac_neighbors_up = NA_real_,
      ppi_frac_neighbors_down = NA_real_,
      ppi_frac_neighbors_unch = NA_real_,
      ppi_weighted_frac_up = NA_real_,
      ppi_weighted_frac_down = NA_real_,
      ppi_weighted_frac_unch = NA_real_
    ))
    next
  }

  # Unweighted fractions
  frac_up <- sum(nbr_labels$label == "up") / n_detected
  frac_down <- sum(nbr_labels$label == "down") / n_detected
  frac_unch <- sum(nbr_labels$label == "unchanged") / n_detected

  # Weighted by STRING confidence
  nbr_edges <- ppi %>%
    filter((from == node & to %in% nbr_labels$STRING_id) |
           (to == node & from %in% nbr_labels$STRING_id)) %>%
    mutate(
      neighbor = ifelse(from == node, to, from),
      weight = combined_score / 1000
    ) %>%
    left_join(label_map %>% dplyr::select(STRING_id, label),
              by = c("neighbor" = "STRING_id"))

  total_weight <- sum(nbr_edges$weight, na.rm = TRUE)
  if (total_weight > 0) {
    w_up <- sum(nbr_edges$weight[nbr_edges$label == "up"], na.rm = TRUE) / total_weight
    w_down <- sum(nbr_edges$weight[nbr_edges$label == "down"], na.rm = TRUE) / total_weight
    w_unch <- sum(nbr_edges$weight[nbr_edges$label == "unchanged"], na.rm = TRUE) / total_weight
  } else {
    w_up <- w_down <- w_unch <- NA_real_
  }

  neighbor_features <- bind_rows(neighbor_features, tibble(
    STRING_id = node,
    ppi_n_detected_neighbors = n_detected,
    ppi_frac_neighbors_up = frac_up,
    ppi_frac_neighbors_down = frac_down,
    ppi_frac_neighbors_unch = frac_unch,
    ppi_weighted_frac_up = w_up,
    ppi_weighted_frac_down = w_down,
    ppi_weighted_frac_unch = w_unch
  ))
}

# Map back to UniProt
neighbor_features <- neighbor_features %>%
  left_join(string_to_uniprot, by = "STRING_id") %>%
  filter(!is.na(UniProt_ID)) %>%
  dplyr::select(-STRING_id)

# --- Compute pathway context features ---
cat("Computing pathway context features...\n")

# Get pathway membership
entrez_ids <- unique(na.omit(entrez_map$ENTREZID))
uniprot_to_entrez <- entrez_map %>%
  dplyr::select(UNIPROT, ENTREZID) %>%
  filter(!is.na(ENTREZID)) %>%
  distinct(UNIPROT, .keep_all = TRUE)

# KEGG pathways from metadata
kegg_pw <- metadata$kegg_pathways
if (nrow(kegg_pw) > 0) {
  # For each protein, compute pathway expression context
  kegg_pw_enriched <- kegg_pw %>%
    left_join(uniprot_to_entrez, by = "ENTREZID") %>%
    filter(!is.na(UNIPROT)) %>%
    left_join(labeled_data %>% dplyr::select(UniProt_ID, label),
              by = c("UNIPROT" = "UniProt_ID"))

  # For each protein-pathway combination, what fraction of pathway members are up/down?
  pathway_stats <- kegg_pw_enriched %>%
    group_by(pathway_id) %>%
    summarise(
      pw_n_members = n(),
      pw_n_labeled = sum(label != "ambiguous", na.rm = TRUE),
      pw_frac_up = sum(label == "up", na.rm = TRUE) / max(pw_n_labeled, 1),
      pw_frac_down = sum(label == "down", na.rm = TRUE) / max(pw_n_labeled, 1),
      .groups = "drop"
    )

  # For each protein, summarize across all its pathways
  pathway_context <- kegg_pw_enriched %>%
    dplyr::select(UNIPROT, pathway_id) %>%
    distinct() %>%
    left_join(pathway_stats, by = "pathway_id") %>%
    group_by(UNIPROT) %>%
    summarise(
      pw_n_pathways = n(),
      pw_max_frac_up = max(pw_frac_up, na.rm = TRUE),
      pw_max_frac_down = max(pw_frac_down, na.rm = TRUE),
      pw_mean_frac_up = mean(pw_frac_up, na.rm = TRUE),
      pw_mean_frac_down = mean(pw_frac_down, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    dplyr::rename(UniProt_ID = UNIPROT) %>%
    mutate(across(starts_with("pw_"), ~ replace(., is.infinite(.), NA_real_)))
} else {
  pathway_context <- tibble(
    UniProt_ID = character(),
    pw_n_pathways = integer(),
    pw_max_frac_up = double(), pw_max_frac_down = double(),
    pw_mean_frac_up = double(), pw_mean_frac_down = double()
  )
}

cat(sprintf("  Pathway context for %d proteins\n", nrow(pathway_context)))

# --- Combine all network features ---
cat("Combining network features...\n")

all_proteins <- tibble(UniProt_ID = unique(de_data$UniProt_ID))

network_features <- all_proteins %>%
  left_join(topo_features, by = "UniProt_ID") %>%
  left_join(neighbor_features, by = "UniProt_ID") %>%
  left_join(pathway_context, by = "UniProt_ID") %>%
  mutate(across(where(is.numeric), ~ replace_na(., 0)))

# --- Save ---
cat("Saving network features...\n")
dir.create(dirname(CONFIG$output_file), recursive = TRUE, showWarnings = FALSE)
write_csv(network_features, CONFIG$output_file)

# Save pathway membership separately (needed for Python CV recomputation)
if (nrow(kegg_pw) > 0) {
  pathway_membership <- kegg_pw %>%
    left_join(uniprot_to_entrez, by = "ENTREZID") %>%
    filter(!is.na(UNIPROT)) %>%
    dplyr::select(UniProt_ID = UNIPROT, pathway_id, source) %>%
    distinct()
  write_csv(pathway_membership, CONFIG$pathway_output)
}

cat(sprintf("\n=== Done ===\n"))
cat(sprintf("  Proteins:         %d\n", nrow(network_features)))
cat(sprintf("  Topology feats:   3 (degree, betweenness, clustering)\n"))
cat(sprintf("  Neighbor feats:   7 (fractions + counts)\n"))
cat(sprintf("  Pathway feats:    5 (pathway context)\n"))
cat(sprintf("  Total features:   %d\n", ncol(network_features) - 1))
cat(sprintf("  Output: %s\n", CONFIG$output_file))

} # end else (PPI data available)
