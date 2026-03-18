#!/usr/bin/env Rscript
# ==============================================================================
# 07_compute_predict_features.R
# Compute features for external proteins (e.g., full human proteome)
# using the same feature pipeline as training but without label-dependent features.
#
# Input:  data/{DATASET}/raw/protein_ids.txt
#         data/{DATASET}/raw/sequences.fasta
#         data/ucec/processed/go_slim_mapping.csv  (for GO term ordering)
# Output: data/{DATASET}/processed/feature_matrix_predict.csv
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(protr)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
  library(STRINGdb)
  library(igraph)
})

# --- Configuration ---
DATASET <- Sys.getenv("DATASET", unset = "human_proteome")
CONFIG <- list(
  ids_file       = sprintf("data/%s/raw/protein_ids.txt", DATASET),
  fasta_file     = sprintf("data/%s/raw/sequences.fasta", DATASET),
  go_mapping     = "data/ucec/processed/go_slim_mapping.csv",
  output_file    = sprintf("data/%s/processed/feature_matrix_predict.csv", DATASET),
  species        = 9606,
  string_version = "12.0",
  string_score_threshold = 400,
  batch_size     = 500
)

cat("=== Phase: Compute Prediction Features ===\n")
cat(sprintf("  Dataset: %s\n", DATASET))

# --- Load protein IDs ---
cat("Loading protein IDs...\n")
protein_ids <- readLines(CONFIG$ids_file)
protein_ids <- protein_ids[nchar(trimws(protein_ids)) > 0]
cat(sprintf("  Loaded %d protein IDs\n", length(protein_ids)))

# --- Parse FASTA sequences ---
cat("Parsing FASTA sequences...\n")
parse_fasta <- function(fasta_path) {
  lines <- readLines(fasta_path)
  ids <- character(); seqs <- character()
  current_id <- NULL; current_seq <- ""
  for (line in lines) {
    if (startsWith(line, ">")) {
      if (!is.null(current_id)) {
        ids <- c(ids, current_id)
        seqs <- c(seqs, current_seq)
      }
      # Parse ">ACCESSION|GENE" or ">ACCESSION"
      header <- sub("^>", "", line)
      current_id <- strsplit(header, "\\|")[[1]][1]
      current_seq <- ""
    } else {
      current_seq <- paste0(current_seq, trimws(line))
    }
  }
  if (!is.null(current_id)) {
    ids <- c(ids, current_id)
    seqs <- c(seqs, current_seq)
  }
  tibble(UniProt_ID = ids, sequence = seqs)
}

seq_data <- parse_fasta(CONFIG$fasta_file)
cat(sprintf("  Parsed %d sequences from FASTA\n", nrow(seq_data)))

# Keep only IDs in our list
seq_data <- seq_data %>% filter(UniProt_ID %in% protein_ids)
cat(sprintf("  Matched %d proteins with sequences\n", nrow(seq_data)))

# ============================================================================
# 1. SEQUENCE FEATURES (protr)
# ============================================================================
cat("\n--- Computing sequence features (protr) ---\n")

clean_sequence <- function(seq) {
  cleaned <- gsub("[^ACDEFGHIKLMNPQRSTVWY]", "", toupper(seq))
  if (nchar(cleaned) < 10) return(NULL)
  cleaned
}

safe_descriptor <- function(seq, func, name) {
  tryCatch(func(seq), error = function(e) NULL)
}

all_seq_features <- list()
n_prots <- nrow(seq_data)

for (i in seq_len(n_prots)) {
  if (i %% 1000 == 0 || i == 1)
    cat(sprintf("  Sequence features: %d/%d\n", i, n_prots))

  uid <- seq_data$UniProt_ID[i]
  seq <- clean_sequence(seq_data$sequence[i])
  if (is.null(seq)) next

  feat <- list(UniProt_ID = uid)

  # AAC (20)
  v <- safe_descriptor(seq, extractAAC, "AAC")
  if (!is.null(v)) { names(v) <- paste0("AAC_", names(v)); feat <- c(feat, as.list(v)) }

  # DC (400)
  v <- safe_descriptor(seq, extractDC, "DC")
  if (!is.null(v)) { names(v) <- paste0("DC_", names(v)); feat <- c(feat, as.list(v)) }

  # CTD (147)
  v <- safe_descriptor(seq, extractCTDC, "CTDC")
  if (!is.null(v)) { names(v) <- paste0("CTDC_", names(v)); feat <- c(feat, as.list(v)) }
  v <- safe_descriptor(seq, extractCTDT, "CTDT")
  if (!is.null(v)) { names(v) <- paste0("CTDT_", names(v)); feat <- c(feat, as.list(v)) }
  v <- safe_descriptor(seq, extractCTDD, "CTDD")
  if (!is.null(v)) { names(v) <- paste0("CTDD_", names(v)); feat <- c(feat, as.list(v)) }

  # CTriad (343)
  v <- safe_descriptor(seq, extractCTriad, "CTriad")
  if (!is.null(v)) { names(v) <- paste0("CTriad_", names(v)); feat <- c(feat, as.list(v)) }

  # QSO (100)
  v <- safe_descriptor(seq, function(s) extractQSO(s, nlag = 30), "QSO")
  if (!is.null(v)) { names(v) <- paste0("QSO_", seq_along(v)); feat <- c(feat, as.list(v)) }

  # APseAAC (30)
  v <- safe_descriptor(seq, function(s) extractAPAAC(s, lambda = 10), "APseAAC")
  if (!is.null(v)) { names(v) <- paste0("APseAAC_", seq_along(v)); feat <- c(feat, as.list(v)) }

  # PAAC (30)
  v <- safe_descriptor(seq, function(s) extractPAAC(s, lambda = 10), "PAAC")
  if (!is.null(v)) { names(v) <- paste0("PseAAC_", seq_along(v)); feat <- c(feat, as.list(v)) }

  all_seq_features[[length(all_seq_features) + 1]] <- feat
}

seq_feat_df <- bind_rows(lapply(all_seq_features, as_tibble))
cat(sprintf("  Sequence features: %d proteins x %d cols\n",
            nrow(seq_feat_df), ncol(seq_feat_df) - 1))

# ============================================================================
# 2. GO-SLIM BINARY MATRIX (matching training ordering exactly)
# ============================================================================
cat("\n--- Computing GO-slim features ---\n")

go_mapping <- read_csv(CONFIG$go_mapping, show_col_types = FALSE)
target_go_terms <- go_mapping$GO  # exact order from training
cat(sprintf("  Using %d GO terms from training mapping\n", length(target_go_terms)))

# Map UniProt -> Entrez
id_lookup <- tryCatch({
  AnnotationDbi::select(org.Hs.eg.db,
    keys = protein_ids,
    keytype = "UNIPROT",
    columns = c("ENTREZID")) %>%
    filter(!is.na(ENTREZID)) %>%
    distinct(UNIPROT, .keep_all = TRUE)
}, error = function(e) {
  cat(sprintf("  Warning: Entrez mapping failed: %s\n", e$message))
  data.frame(UNIPROT = character(), ENTREZID = character())
})
cat(sprintf("  Mapped %d proteins to Entrez IDs\n", nrow(id_lookup)))

# Get GO annotations
go_annots <- tryCatch({
  AnnotationDbi::select(org.Hs.eg.db,
    keys = unique(id_lookup$ENTREZID),
    keytype = "ENTREZID",
    columns = "GO") %>%
    filter(!is.na(GO)) %>%
    dplyr::select(ENTREZID, GO)
}, error = function(e) {
  data.frame(ENTREZID = character(), GO = character())
})

# Build binary matrix with EXACT same columns as training
all_entrez <- unique(id_lookup$ENTREZID)
go_slim_mat <- matrix(0L, nrow = length(all_entrez), ncol = length(target_go_terms),
                       dimnames = list(all_entrez, target_go_terms))

for (i in seq_len(nrow(go_annots))) {
  eid <- go_annots$ENTREZID[i]
  goid <- go_annots$GO[i]
  if (eid %in% rownames(go_slim_mat) && goid %in% colnames(go_slim_mat)) {
    go_slim_mat[eid, goid] <- 1L
  }
}

# Rename columns to GOslim_N (same order as training)
go_slim_df <- as_tibble(go_slim_mat, rownames = "ENTREZID")
colnames(go_slim_df)[2:ncol(go_slim_df)] <- paste0("GOslim_", seq_along(target_go_terms))

go_slim_df <- go_slim_df %>%
  left_join(id_lookup %>% dplyr::select(UNIPROT, ENTREZID), by = "ENTREZID") %>%
  dplyr::select(UniProt_ID = UNIPROT, starts_with("GOslim_")) %>%
  filter(!is.na(UniProt_ID)) %>%
  distinct(UniProt_ID, .keep_all = TRUE)

cat(sprintf("  GO-slim features: %d proteins x %d terms\n",
            nrow(go_slim_df), ncol(go_slim_df) - 1))

# ============================================================================
# 3. STRING PPI TOPOLOGY FEATURES (degree, clustering, betweenness)
# ============================================================================
cat("\n--- Computing STRING PPI topology features ---\n")

# Map to gene symbols for STRING lookup
symbol_map <- tryCatch({
  AnnotationDbi::select(org.Hs.eg.db,
    keys = unique(id_lookup$ENTREZID),
    keytype = "ENTREZID",
    columns = "SYMBOL") %>%
    filter(!is.na(SYMBOL)) %>%
    distinct(ENTREZID, .keep_all = TRUE) %>%
    left_join(id_lookup, by = "ENTREZID")
}, error = function(e) {
  data.frame(ENTREZID = character(), SYMBOL = character(), UNIPROT = character())
})

string_db <- STRINGdb$new(
  version = CONFIG$string_version,
  species = CONFIG$species,
  score_threshold = CONFIG$string_score_threshold
)

# Map in batches to avoid timeouts
n_batches <- ceiling(nrow(symbol_map) / CONFIG$batch_size)
string_mapped_list <- list()

for (b in seq_len(n_batches)) {
  start_i <- (b - 1) * CONFIG$batch_size + 1
  end_i <- min(b * CONFIG$batch_size, nrow(symbol_map))
  batch <- symbol_map[start_i:end_i, ]

  cat(sprintf("  STRING mapping batch %d/%d (%d proteins)...\n",
              b, n_batches, nrow(batch)))

  mapped <- tryCatch(
    string_db$map(data.frame(gene_symbol = batch$SYMBOL,
                              UniProt_ID = batch$UNIPROT),
                  "gene_symbol", removeUnmappedRows = TRUE),
    error = function(e) { cat(sprintf("    Batch failed: %s\n", e$message)); NULL }
  )
  if (!is.null(mapped)) string_mapped_list[[b]] <- mapped
}

string_mapped <- bind_rows(string_mapped_list)
cat(sprintf("  Mapped %d proteins to STRING\n", nrow(string_mapped)))

# Get interactions and build graph
if (nrow(string_mapped) > 0) {
  string_ids <- unique(string_mapped$STRING_id)
  ppi <- string_db$get_interactions(string_ids)
  cat(sprintf("  Retrieved %d PPI interactions\n", nrow(ppi)))

  if (nrow(ppi) > 0) {
    g <- graph_from_data_frame(
      ppi %>% dplyr::select(from, to), directed = FALSE,
      vertices = unique(c(ppi$from, ppi$to))
    )
    E(g)$weight <- ppi$combined_score / 1000

    topo <- tibble(
      STRING_id = V(g)$name,
      ppi_degree = degree(g),
      ppi_betweenness = betweenness(g, normalized = TRUE),
      ppi_clustering = replace_na(transitivity(g, type = "local"), 0)
    )

    string_to_uniprot <- string_mapped %>%
      dplyr::select(UniProt_ID, STRING_id) %>% distinct()

    topo_df <- topo %>%
      left_join(string_to_uniprot, by = "STRING_id") %>%
      filter(!is.na(UniProt_ID)) %>%
      dplyr::select(UniProt_ID, ppi_degree, ppi_betweenness, ppi_clustering) %>%
      distinct(UniProt_ID, .keep_all = TRUE)
  } else {
    topo_df <- tibble(UniProt_ID = character(), ppi_degree = integer(),
                       ppi_betweenness = double(), ppi_clustering = double())
  }
} else {
  topo_df <- tibble(UniProt_ID = character(), ppi_degree = integer(),
                     ppi_betweenness = double(), ppi_clustering = double())
}

cat(sprintf("  Topology features for %d proteins\n", nrow(topo_df)))

# ============================================================================
# 4. DETECTABILITY / PHYSICOCHEMICAL FEATURES
# ============================================================================
cat("\n--- Computing detectability features ---\n")

hydropathy <- c(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5,
                G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8,
                P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)

aa_mw <- c(A=89.09, R=174.20, N=132.12, D=133.10, C=121.16, E=147.13,
           Q=146.15, G=75.03, H=155.16, I=131.17, L=131.17, K=146.19,
           M=149.21, F=165.19, P=115.13, S=105.09, T=119.12, W=204.23,
           Y=181.19, V=117.15)
water_mw <- 18.015

calc_pI <- function(sequence) {
  if (is.na(sequence) || nchar(sequence) < 1) return(NA_real_)
  aa <- strsplit(sequence, "")[[1]]
  n_K <- sum(aa == "K"); n_R <- sum(aa == "R"); n_H <- sum(aa == "H")
  n_D <- sum(aa == "D"); n_E <- sum(aa == "E")
  n_C <- sum(aa == "C"); n_Y <- sum(aa == "Y")
  lo <- 0; hi <- 14
  for (iter in 1:100) {
    pH <- (lo + hi) / 2
    pos <- n_K/(1+10^(pH-10.5)) + n_R/(1+10^(pH-12.4)) +
           n_H/(1+10^(pH-6.0)) + 1/(1+10^(pH-9.69))
    neg <- n_D/(1+10^(3.65-pH)) + n_E/(1+10^(4.25-pH)) +
           n_C/(1+10^(8.18-pH)) + n_Y/(1+10^(10.07-pH)) + 1/(1+10^(2.34-pH))
    net <- sum(pos) - sum(neg)
    if (abs(net) < 0.001) break
    if (net > 0) lo <- pH else hi <- pH
  }
  pH
}

det_list <- list()
for (i in seq_len(nrow(seq_data))) {
  if (i %% 2000 == 0 || i == 1)
    cat(sprintf("  Detectability: %d/%d\n", i, nrow(seq_data)))

  uid <- seq_data$UniProt_ID[i]
  raw_seq <- seq_data$sequence[i]
  if (is.na(raw_seq) || nchar(raw_seq) < 5) next

  aa <- strsplit(toupper(raw_seq), "")[[1]]
  std_aa <- aa[aa %in% names(hydropathy)]
  slen <- length(aa)

  # Tryptic peptides (cleave after K/R not before P)
  matches <- gregexpr("[KR](?!P)", raw_seq, perl = TRUE)[[1]]
  n_tryp <- if (matches[1] == -1L) 1L else length(matches) + 1L

  # GRAVY
  scores <- hydropathy[std_aa]
  gravy <- if (length(scores) > 0) mean(scores, na.rm = TRUE) else NA_real_

  # MW
  mw_vals <- aa_mw[std_aa]
  mw <- if (length(mw_vals) > 0) sum(mw_vals, na.rm = TRUE) - (length(std_aa)-1)*water_mw else NA_real_

  det_list[[length(det_list) + 1]] <- tibble(
    UniProt_ID          = uid,
    det_seq_length      = slen,
    det_log_mw          = if (!is.na(mw) && mw > 0) log10(mw) else NA_real_,
    det_pI              = calc_pI(raw_seq),
    det_charge_pH7      = NA_real_,  # placeholder, compute below
    det_gravy           = gravy,
    det_n_tryptic       = n_tryp,
    det_tryptic_density = n_tryp / (slen / 100),
    det_frac_aromatic   = sum(std_aa %in% c("F","W","Y")) / max(length(std_aa),1),
    det_frac_basic      = sum(std_aa %in% c("K","R","H")) / max(length(std_aa),1),
    det_frac_hydrophobic = sum(std_aa %in% c("A","V","I","L","M","F","W","P")) / max(length(std_aa),1),
    det_is_membrane     = 0L  # default; no subcellular location data for bulk
  )
}

det_df <- bind_rows(det_list)

# Compute charge at pH 7
det_df$det_charge_pH7 <- map_dbl(seq_data$sequence[match(det_df$UniProt_ID, seq_data$UniProt_ID)], function(s) {
  if (is.na(s)) return(NA_real_)
  aa <- strsplit(s, "")[[1]]; pH <- 7.0
  pos <- sum(aa=="K")/(1+10^(pH-10.5)) + sum(aa=="R")/(1+10^(pH-12.4)) +
         sum(aa=="H")/(1+10^(pH-6.0)) + 1/(1+10^(pH-9.69))
  neg <- sum(aa=="D")/(1+10^(3.65-pH)) + sum(aa=="E")/(1+10^(4.25-pH)) +
         sum(aa=="C")/(1+10^(8.18-pH)) + sum(aa=="Y")/(1+10^(10.07-pH)) + 1/(1+10^(2.34-pH))
  pos - neg
})

cat(sprintf("  Detectability features: %d proteins x %d cols\n",
            nrow(det_df), ncol(det_df) - 1))

# ============================================================================
# 5. MERGE ALL FEATURES
# ============================================================================
cat("\n--- Merging all feature sets ---\n")

feature_matrix <- seq_feat_df %>%
  full_join(go_slim_df, by = "UniProt_ID") %>%
  full_join(topo_df, by = "UniProt_ID") %>%
  full_join(det_df, by = "UniProt_ID")

# Fill NAs in topology with 0 (unmapped proteins have no PPI data)
feature_matrix <- feature_matrix %>%
  mutate(
    ppi_degree      = replace_na(ppi_degree, 0),
    ppi_betweenness = replace_na(ppi_betweenness, 0),
    ppi_clustering  = replace_na(ppi_clustering, 0)
  )

# Fill NA GO-slim columns with 0 (unmapped proteins)
go_cols <- colnames(feature_matrix)[grepl("^GOslim_", colnames(feature_matrix))]
for (col in go_cols) {
  feature_matrix[[col]] <- replace_na(feature_matrix[[col]], 0L)
}

cat(sprintf("  Final matrix: %d proteins x %d features\n",
            nrow(feature_matrix), ncol(feature_matrix) - 1))

# --- Save ---
cat("Saving prediction feature matrix...\n")
dir.create(dirname(CONFIG$output_file), recursive = TRUE, showWarnings = FALSE)
write_csv(feature_matrix, CONFIG$output_file)

cat(sprintf("\n=== Done ===\n"))
cat(sprintf("  Proteins:  %d\n", nrow(feature_matrix)))
cat(sprintf("  Features:  %d\n", ncol(feature_matrix) - 1))
cat(sprintf("  Output:    %s\n", CONFIG$output_file))
