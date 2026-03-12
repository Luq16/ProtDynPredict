#!/usr/bin/env Rscript
# ==============================================================================
# 01_fetch_protein_data.R
# Fetch protein metadata from UniProt, STRING, and pathway databases
# Input:  data/raw/de_results.csv (UniProt_ID, log2FC, adj_pvalue)
# Output: data/processed/protein_metadata.rds
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(UniProt.ws)
  library(biomaRt)
  library(STRINGdb)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
})

# --- Configuration ---
DATASET <- Sys.getenv("DATASET", unset = "ucec")
CONFIG <- list(
  input_file      = sprintf("data/%s/raw/de_results.csv", DATASET),
  output_file     = sprintf("data/%s/processed/protein_metadata.rds", DATASET),
  ppi_output      = sprintf("data/%s/processed/ppi_network.rds", DATASET),
  species         = 9606,
  string_version  = "12.0",
  string_score_threshold = 400
)

cat("=== Phase 1.1: Fetch Protein Data ===\n")

# --- Load DE results ---
cat("Loading DE results...\n")
de_data <- read_csv(CONFIG$input_file, show_col_types = FALSE)

required_cols <- c("UniProt_ID", "log2FC", "adj_pvalue")
missing_cols <- setdiff(required_cols, colnames(de_data))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "),
       "\nExpected: UniProt_ID, log2FC, adj_pvalue")
}

uniprot_ids <- unique(de_data$UniProt_ID)
cat(sprintf("  Found %d unique proteins\n", length(uniprot_ids)))

# --- Fetch sequences and annotations from UniProt ---
cat("Fetching protein data from UniProt...\n")

fetch_uniprot_data <- function(ids, batch_size = 100) {
  up <- UniProt.ws(taxId = CONFIG$species)

  available_keytypes <- keytypes(up)
  keytype_to_use <- if ("UniProtKB" %in% available_keytypes) {
    "UniProtKB"
  } else if ("UNIPROTKB" %in% available_keytypes) {
    "UNIPROTKB"
  } else {
    kts <- available_keytypes[grepl("uniprot|entry", available_keytypes, ignore.case = TRUE)]
    if (length(kts) > 0) kts[1] else available_keytypes[1]
  }
  cat(sprintf("  Using keytype: %s\n", keytype_to_use))

  # Detect column names (varies by UniProt.ws version)
  available_cols <- columns(up)

  # Build column list from what's available
  col_candidates <- list(
    accession = c("accession", "Entry"),
    gene_names = c("gene_names", "Gene.Names", "GENES"),
    protein_name = c("protein_name", "Protein.names", "PROTEIN-NAMES"),
    sequence = c("sequence", "Sequence", "SEQUENCE"),
    length = c("length", "Length", "LENGTH"),
    go_id = c("go_id", "Gene.Ontology.IDs", "GO-ID"),
    go_p = c("go_p", "Gene.Ontology..biological.process.", "GO"),
    go_f = c("go_f", "Gene.Ontology..molecular.function."),
    go_c = c("go_c", "Gene.Ontology..cellular.component."),
    cc_subcellular_location = c("cc_subcellular_location", "Subcellular.location..CC.", "SUBCELLULAR-LOCATIONS"),
    mass = c("mass", "Mass", "MASS"),
    xref_pfam = c("xref_pfam", "Pfam", "INTERPRO")
  )

  columns_to_fetch <- c()
  col_name_map <- list()  # maps actual column name -> canonical name
  for (canonical in names(col_candidates)) {
    for (candidate in col_candidates[[canonical]]) {
      if (candidate %in% available_cols) {
        columns_to_fetch <- c(columns_to_fetch, candidate)
        col_name_map[[candidate]] <- canonical
        break
      }
    }
  }
  cat(sprintf("  Fetching columns: %s\n", paste(columns_to_fetch, collapse = ", ")))

  all_results <- list()
  n_batches <- ceiling(length(ids) / batch_size)

  for (i in seq_len(n_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, length(ids))
    batch_ids <- ids[start_idx:end_idx]

    cat(sprintf("  Batch %d/%d (%d proteins)...\n", i, n_batches, length(batch_ids)))

    tryCatch({
      # Use UniProt.ws::select explicitly to avoid AnnotationDbi conflict
      result <- UniProt.ws::select(up,
                       keys = batch_ids,
                       keytype = keytype_to_use,
                       columns = columns_to_fetch)
      all_results[[i]] <- result
    }, error = function(e) {
      cat(sprintf("  Warning: Batch %d failed: %s\n", i, e$message))
      all_results[[i]] <<- NULL
    })

    if (i < n_batches) Sys.sleep(1)
  }

  result_df <- bind_rows(all_results)

  # Rename all returned columns to canonical names
  # UniProt.ws may return different names than what was requested
  all_renames <- c(
    "From" = "accession", "Entry" = "accession",
    "Gene.Names" = "gene_names", "GENES" = "gene_names",
    "Protein.names" = "protein_name", "PROTEIN-NAMES" = "protein_name",
    "Sequence" = "sequence", "SEQUENCE" = "sequence",
    "Length" = "length", "LENGTH" = "length",
    "Gene.Ontology.IDs" = "go_id", "GO-ID" = "go_id",
    "Gene.Ontology..biological.process." = "go_p",
    "Gene.Ontology..molecular.function." = "go_f",
    "Gene.Ontology..cellular.component." = "go_c",
    "Subcellular.location..CC." = "cc_subcellular_location",
    "SUBCELLULAR-LOCATIONS" = "cc_subcellular_location",
    "Mass" = "mass", "MASS" = "mass",
    "Pfam" = "xref_pfam", "INTERPRO" = "xref_pfam"
  )
  for (old_name in names(all_renames)) {
    new_name <- all_renames[old_name]
    if (old_name %in% colnames(result_df) && !new_name %in% colnames(result_df)) {
      colnames(result_df)[colnames(result_df) == old_name] <- new_name
    }
  }

  cat(sprintf("  Final columns: %s\n", paste(colnames(result_df), collapse = ", ")))

  result_df
}

uniprot_data <- fetch_uniprot_data(uniprot_ids)
cat(sprintf("  Retrieved data for %d proteins\n", n_distinct(uniprot_data$accession)))

# --- Parse GO annotations ---
cat("Parsing GO annotations...\n")

parse_go_terms <- function(go_string) {
  if (is.na(go_string) || go_string == "") return(character(0))
  trimws(unlist(strsplit(go_string, ";")))
}

# Use available GO columns
go_cols <- intersect(c("go_p", "go_f", "go_c", "go_id"), colnames(uniprot_data))
if (length(go_cols) > 0) {
  go_annotations <- uniprot_data %>%
    dplyr::select(accession, dplyr::any_of(c("go_p", "go_f", "go_c"))) %>%
    distinct()

  # Fill missing GO columns
  for (col in c("go_p", "go_f", "go_c")) {
    if (!col %in% colnames(go_annotations)) {
      go_annotations[[col]] <- NA_character_
    }
  }

  go_annotations <- go_annotations %>%
    mutate(
      GO_BP = map(go_p, parse_go_terms),
      GO_MF = map(go_f, parse_go_terms),
      GO_CC = map(go_c, parse_go_terms)
    ) %>%
    dplyr::select(accession, GO_BP, GO_MF, GO_CC)
} else {
  cat("  Warning: No GO columns available, creating empty annotations\n")
  go_annotations <- tibble(
    accession = unique(uniprot_data$accession),
    GO_BP = list(character(0)),
    GO_MF = list(character(0)),
    GO_CC = list(character(0))
  )
}

# --- Fetch PPI network from STRING ---
cat("Fetching PPI network from STRING...\n")

string_db <- STRINGdb$new(
  version = CONFIG$string_version,
  species = CONFIG$species,
  score_threshold = CONFIG$string_score_threshold
)

# Map to STRING IDs via gene symbols
entrez_map_early <- tryCatch({
  AnnotationDbi::select(org.Hs.eg.db,
         keys = uniprot_ids,
         keytype = "UNIPROT",
         columns = c("ENTREZID", "SYMBOL")) %>%
    distinct(UNIPROT, .keep_all = TRUE) %>%
    filter(!is.na(SYMBOL))
}, error = function(e) {
  data.frame(UNIPROT = character(), ENTREZID = character(), SYMBOL = character())
})

if (nrow(entrez_map_early) > 0) {
  id_mapping <- data.frame(
    UniProt_ID = entrez_map_early$UNIPROT,
    gene_symbol = entrez_map_early$SYMBOL
  )
  string_mapped <- tryCatch(
    string_db$map(id_mapping, "gene_symbol", removeUnmappedRows = TRUE),
    error = function(e) {
      cat(sprintf("  Gene symbol mapping failed, trying UniProt IDs: %s\n", e$message))
      string_db$map(data.frame(UniProt_ID = uniprot_ids), "UniProt_ID",
                     removeUnmappedRows = TRUE)
    }
  )
  if (!"UniProt_ID" %in% colnames(string_mapped) && "UNIPROT" %in% colnames(id_mapping)) {
    string_mapped$UniProt_ID <- id_mapping$UniProt_ID[
      match(string_mapped$gene_symbol, id_mapping$gene_symbol)]
  }
} else {
  id_mapping <- data.frame(UniProt_ID = uniprot_ids)
  string_mapped <- string_db$map(id_mapping, "UniProt_ID", removeUnmappedRows = TRUE)
}
cat(sprintf("  Mapped %d/%d proteins to STRING\n",
            nrow(string_mapped), length(uniprot_ids)))

# Get interactions for mapped proteins
if (nrow(string_mapped) > 0) {
  string_ids <- string_mapped$STRING_id
  ppi_interactions <- string_db$get_interactions(string_ids)
  cat(sprintf("  Retrieved %d interactions\n", nrow(ppi_interactions)))
} else {
  ppi_interactions <- data.frame(from = character(), to = character(),
                                  combined_score = numeric())
  cat("  Warning: No STRING mappings found\n")
}

# --- Map UniProt to Entrez for pathway analysis ---
cat("Mapping UniProt IDs to Entrez Gene IDs...\n")

entrez_map <- tryCatch({
  AnnotationDbi::select(org.Hs.eg.db,
         keys = uniprot_ids,
         keytype = "UNIPROT",
         columns = c("ENTREZID", "SYMBOL")) %>%
    distinct()
}, error = function(e) {
  cat(sprintf("  Warning: Entrez mapping failed: %s\n", e$message))
  data.frame(UNIPROT = character(), ENTREZID = character(), SYMBOL = character())
})

cat(sprintf("  Mapped %d proteins to Entrez IDs\n", n_distinct(entrez_map$UNIPROT)))

# --- Fetch pathway membership ---
cat("Fetching pathway membership (KEGG)...\n")

entrez_ids <- unique(na.omit(entrez_map$ENTREZID))

kegg_pathways <- tryCatch({
  AnnotationDbi::select(org.Hs.eg.db,
         keys = entrez_ids,
         keytype = "ENTREZID",
         columns = "PATH") %>%
    filter(!is.na(PATH)) %>%
    dplyr::rename(pathway_id = PATH) %>%
    mutate(pathway_id = paste0("hsa", pathway_id),
           source = "KEGG")
}, error = function(e) {
  cat(sprintf("  Warning: KEGG fetch failed: %s\n", e$message))
  data.frame(ENTREZID = character(), pathway_id = character(), source = character())
})

cat(sprintf("  Found %d KEGG pathway annotations\n", nrow(kegg_pathways)))

# --- Parse Pfam domains ---
cat("Parsing Pfam domain annotations...\n")

pfam_data <- uniprot_data %>%
  dplyr::select(accession, dplyr::any_of("xref_pfam")) %>%
  distinct()

if ("xref_pfam" %in% colnames(pfam_data)) {
  pfam_data <- pfam_data %>%
    mutate(pfam_domains = map(xref_pfam, ~ {
      if (is.na(.x) || .x == "") return(character(0))
      trimws(unlist(strsplit(.x, ";")))
    }))
} else {
  pfam_data$xref_pfam <- NA_character_
  pfam_data$pfam_domains <- list(character(0))
}

# --- Calculate basic protein properties ---
cat("Calculating basic protein properties...\n")

protein_properties <- uniprot_data %>%
  dplyr::select(accession, dplyr::any_of(c("sequence", "length", "mass", "cc_subcellular_location"))) %>%
  distinct()

# Ensure columns exist with correct types
for (col in c("sequence", "length", "mass", "cc_subcellular_location")) {
  if (!col %in% colnames(protein_properties)) {
    protein_properties[[col]] <- NA_character_
  }
}
# Ensure character type for string columns
protein_properties$cc_subcellular_location <- as.character(protein_properties$cc_subcellular_location)
protein_properties$sequence <- as.character(protein_properties$sequence)

protein_properties <- protein_properties %>%
  mutate(
    molecular_weight = as.numeric(mass),
    seq_length = as.numeric(length),
    n_tryptic_peptides = map_int(sequence, ~ {
      if (is.na(.x)) return(0L)
      matches <- gregexpr("[KR](?!P)", .x, perl = TRUE)[[1]]
      if (matches[1] == -1L) return(1L)
      length(matches) + 1L
    }),
    gravy_score = map_dbl(sequence, ~ {
      if (is.na(.x)) return(NA_real_)
      aa <- strsplit(.x, "")[[1]]
      hydropathy <- c(
        A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5,
        G=-0.4, H=-3.2, I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8,
        P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2
      )
      scores <- hydropathy[aa]
      scores <- scores[!is.na(scores)]
      if (length(scores) == 0) return(NA_real_)
      mean(scores)
    }),
    is_membrane = str_detect(
      replace_na(cc_subcellular_location, ""),
      regex("membrane|transmembrane", ignore_case = TRUE)
    )
  )

# --- Assemble metadata object ---
cat("Assembling metadata...\n")

metadata <- list(
  de_data = de_data,
  uniprot_data = uniprot_data,
  go_annotations = go_annotations,
  protein_properties = protein_properties,
  pfam_data = pfam_data,
  entrez_map = entrez_map,
  kegg_pathways = kegg_pathways,
  string_mapping = string_mapped,
  ppi_interactions = ppi_interactions,
  config = CONFIG
)

# --- Save outputs ---
cat("Saving outputs...\n")
dir.create(dirname(CONFIG$output_file), recursive = TRUE, showWarnings = FALSE)
saveRDS(metadata, CONFIG$output_file)
saveRDS(ppi_interactions, CONFIG$ppi_output)

cat(sprintf("\n=== Done ===\n"))
cat(sprintf("  Proteins in DE data:    %d\n", length(uniprot_ids)))
cat(sprintf("  UniProt data retrieved: %d\n", n_distinct(uniprot_data$accession)))
cat(sprintf("  STRING mapped:          %d\n", nrow(string_mapped)))
cat(sprintf("  PPI interactions:       %d\n", nrow(ppi_interactions)))
cat(sprintf("  Entrez mapped:          %d\n", n_distinct(entrez_map$UNIPROT)))
cat(sprintf("  Output: %s\n", CONFIG$output_file))
