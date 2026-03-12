#!/usr/bin/env Rscript
# ==============================================================================
# install_dependencies.R
# Install all required R packages for the ProtDynPredict pipeline
# Usage: Rscript R/install_dependencies.R
# ==============================================================================

cat("=== Installing ProtDynPredict R Dependencies ===\n\n")

# --- CRAN packages ---
cran_packages <- c("tidyverse", "protr", "igraph", "reticulate")

cat("Installing CRAN packages...\n")
for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    cat(sprintf("  %s already installed\n", pkg))
  }
}

# --- Bioconductor packages ---
bioc_packages <- c(
  "GOSemSim", "org.Hs.eg.db", "STRINGdb", "biomaRt",
  "UniProt.ws", "AnnotationDbi", "GO.db",
  "clusterProfiler", "ReactomePA",
  "ExperimentHub", "DEqMS", "limma"
)

cat("\nInstalling Bioconductor packages...\n")
if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org")
}

for (pkg in bioc_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  Installing %s...\n", pkg))
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  } else {
    cat(sprintf("  %s already installed\n", pkg))
  }
}

# --- Verify ---
cat("\n=== Verification ===\n")
all_packages <- c(cran_packages, bioc_packages)
status <- sapply(all_packages, requireNamespace, quietly = TRUE)
for (i in seq_along(all_packages)) {
  cat(sprintf("  %-20s %s\n", all_packages[i],
              ifelse(status[i], "OK", "FAILED")))
}

if (all(status)) {
  cat("\nAll packages installed successfully.\n")
  cat("Run next:\n")
  cat("  Option A (easiest):      Rscript R/fetch_real_data_deqms.R\n")
  cat("  Option B (tumor/normal): Rscript R/fetch_real_data_cptac.R\n")
} else {
  cat("\nSome packages failed to install. Check errors above.\n")
}
