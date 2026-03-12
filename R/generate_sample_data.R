#!/usr/bin/env Rscript
# ==============================================================================
# generate_sample_data.R
# Generate a realistic sample DE proteomics dataset for testing the pipeline
# Uses real human UniProt IDs from well-characterized cancer proteomics
# Usage: Rscript R/generate_sample_data.R
# Output: data/raw/de_results.csv
# ==============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
})

cat("=== Generating Sample DE Proteomics Data ===\n")

set.seed(42)

# Real human UniProt IDs — commonly detected in cancer proteomics
# These are well-annotated proteins with known STRING interactions and GO terms
uniprot_ids <- c(
  # Upregulated in many cancers (oncoproteins, proliferation markers)
  "P04637",  # TP53
  "P38936",  # CDKN1A (p21)
  "P06493",  # CDK1
  "P24941",  # CDK2
  "P30304",  # CDC25A
  "P29350",  # PTPN6 (SHP-1)
  "P00533",  # EGFR
  "P04049",  # RAF1
  "P01112",  # HRAS
  "P01116",  # KRAS
  "P15056",  # BRAF
  "Q07817",  # BCL2L1 (BCL-XL)
  "P10415",  # BCL2
  "Q07812",  # BAX
  "O14727",  # APAF1
  "P42574",  # CASP3
  "P55957",  # BID
  "P42771",  # CDKN2A (p16)
  "Q00987",  # MDM2
  "P31749",  # AKT1
  "P31751",  # AKT2
  "P42336",  # PIK3CA
  "P27986",  # PIK3R1
  "Q92934",  # BAD
  "P28482",  # MAPK1 (ERK2)
  "P27361",  # MAPK3 (ERK1)
  "Q16539",  # MAPK14 (p38)
  "P45983",  # MAPK8 (JNK1)
  "P46108",  # CRK
  "Q13315",  # ATM

  # Metabolic enzymes (often detected, various changes)
  "P00338",  # LDHA
  "P07195",  # LDHB
  "P04406",  # GAPDH
  "P60174",  # TPI1
  "P06733",  # ENO1
  "P14618",  # PKM
  "P11413",  # G6PD
  "O75874",  # IDH1
  "P48735",  # IDH2
  "Q02218",  # OGDH
  "P10809",  # HSPD1 (HSP60)
  "P08238",  # HSP90AB1
  "P07900",  # HSP90AA1
  "P11142",  # HSPA8 (HSC70)
  "P08107",  # HSPA1A (HSP70)

  # Cytoskeleton / structural (often unchanged)
  "P60709",  # ACTB
  "P63261",  # ACTG1
  "P68371",  # TUBB4B
  "P07437",  # TUBB
  "P68363",  # TUBA1B
  "Q71U36",  # TUBA1A
  "P06753",  # TPM3
  "P09493",  # TPM1
  "P35579",  # MYH9
  "P35580",  # MYH10

  # Ribosomal proteins (often unchanged/housekeeping)
  "P62258",  # YWHAE
  "P31946",  # YWHAB
  "P27348",  # YWHAQ
  "P63104",  # YWHAZ
  "P61981",  # YWHAG
  "P62753",  # RPS6
  "P23396",  # RPS3
  "P62263",  # RPS14
  "P46781",  # RPS9
  "P62829",  # RPL23

  # Downregulated in cancer contexts
  "P01023",  # A2M
  "P02768",  # ALB
  "P02787",  # TF (transferrin)
  "P00450",  # CP (ceruloplasmin)
  "P02751",  # FN1 (fibronectin)
  "P01024",  # C3
  "P08603",  # CFH
  "P07358",  # C8B
  "P00747",  # PLG
  "P02675",  # FGB

  # DNA repair / cell cycle
  "P51587",  # BRCA2
  "P38398",  # BRCA1
  "O43681",  # ASNA1
  "P12004",  # PCNA
  "P09874",  # PARP1
  "Q9Y6D9",  # MAD1L1
  "O00744",  # SLC11A2
  "P11388",  # TOP2A
  "P13010",  # XRCC5 (KU80)
  "P12956",  # XRCC6 (KU70)

  # Immune / signaling
  "P05362",  # ICAM1
  "P16284",  # PECAM1
  "P19320",  # VCAM1
  "P01375",  # TNF
  "P05231",  # IL6
  "P10145",  # CXCL8 (IL-8)
  "P60568",  # IL2
  "P01579",  # IFNG
  "P22301",  # IL10
  "P01589"   # IL2RA
)

n_proteins <- length(uniprot_ids)
cat(sprintf("  Generating DE results for %d proteins\n", n_proteins))

# Simulate realistic DE results
# ~15% upregulated, ~10% downregulated, ~60% unchanged, ~15% ambiguous
de_results <- tibble(
  UniProt_ID = uniprot_ids,
  group = c(
    rep("signaling", 30),
    rep("metabolic", 15),
    rep("structural", 10),
    rep("ribosomal", 10),
    rep("secreted", 10),
    rep("dna_repair", 10),
    rep("immune", 10)
  )
) %>%
  mutate(
    # Assign expression pattern based on biology
    true_state = case_when(
      # Signaling proteins: mixed (cancer context)
      group == "signaling" ~ sample(c("up", "up", "down", "unchanged", "unchanged"),
                                     n(), replace = TRUE),
      # Metabolic: mostly up (Warburg effect)
      group == "metabolic" ~ sample(c("up", "up", "unchanged", "unchanged", "down"),
                                     n(), replace = TRUE),
      # Structural: mostly unchanged
      group == "structural" ~ sample(c("unchanged", "unchanged", "unchanged", "up", "down"),
                                      n(), replace = TRUE),
      # Ribosomal: mostly unchanged
      group == "ribosomal" ~ sample(c("unchanged", "unchanged", "unchanged", "up", "down"),
                                     n(), replace = TRUE),
      # Secreted: mostly down in solid tumors
      group == "secreted" ~ sample(c("down", "down", "down", "unchanged", "unchanged"),
                                    n(), replace = TRUE),
      # DNA repair: mixed
      group == "dna_repair" ~ sample(c("up", "up", "unchanged", "unchanged", "down"),
                                      n(), replace = TRUE),
      # Immune: mixed
      group == "immune" ~ sample(c("up", "down", "unchanged", "up", "down"),
                                  n(), replace = TRUE),
      TRUE ~ "unchanged"
    ),
    # Generate log2FC based on true state
    log2FC = case_when(
      true_state == "up" ~ rnorm(n(), mean = 2.0, sd = 0.8),
      true_state == "down" ~ rnorm(n(), mean = -1.8, sd = 0.7),
      true_state == "unchanged" ~ rnorm(n(), mean = 0, sd = 0.3)
    ),
    # Generate p-values (significant for true DE, non-significant for unchanged)
    adj_pvalue = case_when(
      true_state %in% c("up", "down") ~ 10^(runif(n(), -8, -1.5)),
      true_state == "unchanged" ~ runif(n(), 0.05, 0.95)
    )
  ) %>%
  # Clip to realistic ranges
  mutate(
    log2FC = pmax(pmin(log2FC, 6), -6),
    adj_pvalue = pmax(adj_pvalue, 1e-10)
  ) %>%
  select(UniProt_ID, log2FC, adj_pvalue)

# --- Save ---
dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
write_csv(de_results, "data/raw/de_results.csv")

cat(sprintf("\n  Output: data/raw/de_results.csv\n"))
cat(sprintf("  Total proteins: %d\n", nrow(de_results)))

# Quick label check
labels <- de_results %>%
  mutate(
    label = case_when(
      log2FC > 1.0 & adj_pvalue < 0.05 ~ "up",
      log2FC < -1.0 & adj_pvalue < 0.05 ~ "down",
      abs(log2FC) < 0.5 & adj_pvalue > 0.20 ~ "unchanged",
      TRUE ~ "ambiguous"
    )
  )

cat("\n  Label distribution with default thresholds:\n")
print(table(labels$label))
cat("\n  Run next: Rscript R/00_run_pipeline.R\n")
