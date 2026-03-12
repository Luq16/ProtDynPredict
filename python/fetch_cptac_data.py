#!/usr/bin/env python3
"""
fetch_cptac_data.py
Download CPTAC tumor vs normal proteomics data and run DE analysis.

Dataset: CPTAC Endometrial Cancer (UCEC)
         Real MS-based proteomics, tumor vs normal adjacent tissue
         103 tumor + 49 normal samples, ~12K proteins

Output: data/raw/de_results.csv (UniProt_ID, log2FC, adj_pvalue)
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Monkey-patch pyranges (cptac imports it but we only need proteomics, not genomic coords)
import types
pyranges_mock = types.ModuleType('pyranges')
pyranges_mock.read_gtf = lambda *a, **k: None
sys.modules['pyranges'] = pyranges_mock

import cptac

# --- Configuration ---
cancer_type = sys.argv[1] if len(sys.argv) > 1 else "ucec"
cancer_type = cancer_type.lower()

CANCER_MAP = {
    "ucec": ("Ucec", "Endometrial"),
    "coad": ("Coad", "Colorectal"),
    "brca": ("Brca", "Breast"),
    "luad": ("Luad", "Lung Adenocarcinoma"),
    "hnscc": ("Hnscc", "Head and Neck"),
    "ov": ("Ov", "Ovarian"),
}

if cancer_type not in CANCER_MAP:
    print(f"Error: unsupported cancer type '{cancer_type}'")
    print(f"  Supported: {', '.join(CANCER_MAP.keys())}")
    sys.exit(1)

cptac_class_name, cancer_label = CANCER_MAP[cancer_type]

MAX_PROTEINS = 3500   # Subset for fast iteration (set to None for all)

print("=" * 60)
print("  FETCHING CPTAC PROTEOMICS DATA")
print("=" * 60)
print(f"  Cancer type: {cancer_type.upper()} ({cancer_label})")
print(f"  Comparison: Tumor vs Normal adjacent tissue")
print(f"  Max proteins: {MAX_PROTEINS or 'all'}")
print()

# --- Download dataset ---
print(f"Downloading CPTAC {cancer_type.upper()} data (first run downloads ~100-200 MB)...")
dataset = getattr(cptac, cptac_class_name)()
print("  Dataset loaded.")

# --- Extract proteomics ---
print("Extracting proteomics quantification...")
try:
    prot_raw = dataset.get_proteomics(source='umich')
except Exception:
    try:
        prot_raw = dataset.get_proteomics(source='harmonized')
    except Exception:
        prot_raw = dataset.get_proteomics()

# Use gene symbols as column names (Database_IDs are Ensembl, not UniProt)
if isinstance(prot_raw.columns, pd.MultiIndex):
    prot_df = prot_raw.copy()
    prot_df.columns = prot_raw.columns.get_level_values(0)
else:
    prot_df = prot_raw.copy()

print(f"  Proteomics matrix: {prot_df.shape[0]} samples x {prot_df.shape[1]} proteins")

# --- Identify tumor vs normal ---
print("Identifying tumor vs normal samples...")
sample_ids = prot_df.index.tolist()
sample_type = np.array([
    "Normal" if str(sid).endswith(".N") or "Normal" in str(sid)
    else "Tumor"
    for sid in sample_ids
])

n_tumor = np.sum(sample_type == "Tumor")
n_normal = np.sum(sample_type == "Normal")
print(f"  Tumor samples:  {n_tumor}")
print(f"  Normal samples: {n_normal}")

# --- Handle duplicate gene columns ---
# Average duplicate gene columns
prot_df = prot_df.T.groupby(level=0).mean().T
print(f"  After merging duplicate genes: {prot_df.shape[1]} proteins")

# --- Filter proteins ---
print("Filtering proteins...")
missing_frac = prot_df.isnull().mean(axis=0)
keep = missing_frac < 0.5
prot_df = prot_df.loc[:, keep]
print(f"  After missing value filter: {prot_df.shape[1]} proteins")

# --- DE analysis (Welch's t-test per protein) ---
print("Running DE analysis (Welch's t-test)...")

tumor_mask = sample_type == "Tumor"
normal_mask = sample_type == "Normal"

results = []
for gene in prot_df.columns:
    vals = prot_df[gene]
    tumor_vals = vals[tumor_mask].dropna()
    normal_vals = vals[normal_mask].dropna()

    if len(tumor_vals) < 3 or len(normal_vals) < 3:
        continue

    # CPTAC data is already log2-ratio normalized
    log2fc = tumor_vals.mean() - normal_vals.mean()
    t_stat, p_val = stats.ttest_ind(tumor_vals, normal_vals, equal_var=False)

    if np.isnan(p_val) or np.isnan(log2fc):
        continue

    results.append({
        "gene_symbol": gene,
        "log2FC": log2fc,
        "pvalue": p_val,
    })

results_df = pd.DataFrame(results)
print(f"  Tested: {len(results_df)} proteins")

# --- Multiple testing correction (BH) ---
from statsmodels.stats.multitest import multipletests
_, adj_pvals, _, _ = multipletests(results_df["pvalue"].values, method="fdr_bh")
results_df["adj_pvalue"] = adj_pvals

# --- Map gene symbols to UniProt IDs ---
print("Mapping gene symbols to UniProt IDs...")
import requests

unique_genes = results_df["gene_symbol"].unique().tolist()
print(f"  Querying UniProt REST API for {len(unique_genes)} gene symbols...")

api_map = {}
batch_size = 50  # Small batches to avoid URL length limits
for i in range(0, len(unique_genes), batch_size):
    batch = unique_genes[i:i + batch_size]
    query = " OR ".join([f"(gene_exact:{g})" for g in batch])
    try:
        resp = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query": f"({query}) AND (organism_id:9606) AND (reviewed:true)",
                "fields": "accession,gene_primary",
                "format": "tsv",
                "size": "500",
            },
            timeout=30,
        )
        if resp.status_code == 200:
            for line in resp.text.strip().split("\n")[1:]:
                parts = line.split("\t")
                if len(parts) >= 2:
                    uid, gene = parts[0], parts[1]
                    if gene in batch and gene not in api_map:
                        api_map[gene] = uid
    except Exception as e:
        print(f"    Warning: API batch failed: {e}")

    if (i // batch_size) % 20 == 0 and i > 0:
        print(f"    Progress: {i}/{len(unique_genes)} genes, {len(api_map)} mapped")

results_df["UniProt_ID"] = results_df["gene_symbol"].map(api_map)
print(f"  Mapped {results_df['UniProt_ID'].notna().sum()}/{len(results_df)} genes to UniProt IDs")

# Drop proteins without UniProt IDs
n_before = len(results_df)
results_df = results_df.dropna(subset=["UniProt_ID"])
print(f"  Dropped {n_before - len(results_df)} proteins without UniProt IDs")

# Keep best result per UniProt ID
results_df = results_df.sort_values("adj_pvalue").drop_duplicates("UniProt_ID", keep="first")
results_df = results_df[["UniProt_ID", "log2FC", "adj_pvalue"]].reset_index(drop=True)
print(f"  Final proteins with UniProt IDs: {len(results_df)}")

# --- Subset for fast iteration ---
if MAX_PROTEINS and len(results_df) > MAX_PROTEINS:
    print(f"\nSubsetting to {MAX_PROTEINS} proteins...")
    np.random.seed(42)

    sig_up = results_df[(results_df["log2FC"] > 0.5) & (results_df["adj_pvalue"] < 0.05)]
    sig_down = results_df[(results_df["log2FC"] < -0.5) & (results_df["adj_pvalue"] < 0.05)]
    unchanged = results_df[(results_df["log2FC"].abs() < 0.25) & (results_df["adj_pvalue"] > 0.20)]
    ambiguous = results_df[
        ~results_df["UniProt_ID"].isin(
            pd.concat([sig_up, sig_down, unchanged])["UniProt_ID"]
        )
    ]

    n_sig = len(sig_up) + len(sig_down)
    n_remaining = max(0, MAX_PROTEINS - n_sig)
    n_unch = min(len(unchanged), max(0, int(n_remaining * 0.7)))
    n_ambig = min(len(ambiguous), max(0, n_remaining - n_unch))

    results_df = pd.concat([
        sig_up,
        sig_down,
        unchanged.sample(n=n_unch, random_state=42),
        ambiguous.sample(n=max(0, n_ambig), random_state=42),
    ]).reset_index(drop=True)

    print(f"  Subset: {len(sig_up)} up, {len(sig_down)} down, "
          f"{n_unch} unchanged, {max(0, n_ambig)} ambiguous = {len(results_df)} total")

# --- Summary ---
print(f"\nDE Summary (CPTAC {cancer_type.upper()}, Tumor vs Normal):")
print(f"  Total proteins:    {len(results_df)}")
print(f"  Upregulated:       {((results_df['log2FC'] > 0.5) & (results_df['adj_pvalue'] < 0.05)).sum()} (log2FC > 0.5, adj_p < 0.05)")
print(f"  Downregulated:     {((results_df['log2FC'] < -0.5) & (results_df['adj_pvalue'] < 0.05)).sum()} (log2FC < -0.5, adj_p < 0.05)")
print(f"  Unchanged:         {((results_df['log2FC'].abs() < 0.25) & (results_df['adj_pvalue'] > 0.20)).sum()} (|log2FC| < 0.25, adj_p > 0.20)")
print(f"  log2FC range:      [{results_df['log2FC'].min():.2f}, {results_df['log2FC'].max():.2f}]")

# --- Save ---
output_dir = Path(f"data/{cancer_type}/raw")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "de_results.csv"
results_df.to_csv(output_path, index=False)

print(f"\n  Output: {output_path}")
print(f"  Next: DATASET={cancer_type} Rscript R/00_run_pipeline.R")
print("\n=== Done ===")
