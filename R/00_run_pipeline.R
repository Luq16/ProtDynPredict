#!/usr/bin/env Rscript
# ==============================================================================
# 00_run_pipeline.R
# Master script to run the full R feature engineering pipeline
# Usage: Rscript R/00_run_pipeline.R
# ==============================================================================

cat("============================================\n")
cat("  ProtDynPredict - Feature Engineering\n")
cat("============================================\n\n")

# --- Check dependencies ---
cat("Checking R package dependencies...\n")
required_packages <- c(
  "tidyverse", "protr", "GOSemSim", "org.Hs.eg.db",
  "STRINGdb", "igraph", "biomaRt", "UniProt.ws",
  "AnnotationDbi", "GO.db", "clusterProfiler", "ReactomePA"
)

missing <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing) > 0) {
  cat("\nMissing packages:\n")
  cat(paste("  -", missing, collapse = "\n"), "\n\n")
  cat("Install with:\n")

  bioc_pkgs <- intersect(missing, c("GOSemSim", "org.Hs.eg.db", "STRINGdb",
                                     "biomaRt", "UniProt.ws", "AnnotationDbi",
                                     "GO.db", "clusterProfiler", "ReactomePA"))
  cran_pkgs <- setdiff(missing, bioc_pkgs)

  if (length(cran_pkgs) > 0) {
    cat(sprintf('  install.packages(c(%s))\n',
                paste0('"', cran_pkgs, '"', collapse = ", ")))
  }
  if (length(bioc_pkgs) > 0) {
    cat('  if (!require("BiocManager")) install.packages("BiocManager")\n')
    cat(sprintf('  BiocManager::install(c(%s))\n',
                paste0('"', bioc_pkgs, '"', collapse = ", ")))
  }
  stop("\nPlease install missing packages before running the pipeline.")
}
cat("  All packages available.\n\n")

# --- Check input data ---
if (!file.exists("data/raw/de_results.csv")) {
  stop("Input file not found: data/raw/de_results.csv\n",
       "  Place your differential expression results there with columns:\n",
       "  UniProt_ID, log2FC, adj_pvalue\n\n",
       "  Or run: Rscript R/generate_sample_data.R")
}

# --- Run pipeline ---
scripts <- c(
  "R/01_fetch_protein_data.R",
  "R/02_sequence_features.R",
  "R/03_go_similarity.R",
  "R/04_network_features.R",
  "R/05_detectability_features.R",
  "R/06_assemble_features.R"
)

start_time <- Sys.time()

for (script in scripts) {
  cat(sprintf("\n>>> Running %s\n", script))
  cat(strrep("-", 60), "\n")

  tryCatch({
    source(script, local = new.env())
  }, error = function(e) {
    cat(sprintf("\n!!! ERROR in %s: %s\n", script, e$message))
    cat("Pipeline stopped. Fix the error and re-run.\n")
    stop(e)
  })

  cat(strrep("-", 60), "\n")
}

elapsed <- difftime(Sys.time(), start_time, units = "mins")
cat(sprintf("\n============================================\n"))
cat(sprintf("  Pipeline complete in %.1f minutes\n", elapsed))
cat(sprintf("  Next: run python/00_validate_premise.py\n"))
cat(sprintf("============================================\n"))
