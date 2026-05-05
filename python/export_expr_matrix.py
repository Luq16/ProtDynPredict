#!/usr/bin/env python3
"""Export raw expression matrix + sample labels from CPTAC for limma in R."""

import sys, warnings, types
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# Monkey-patch pyranges
pyranges_mock = types.ModuleType('pyranges')
pyranges_mock.read_gtf = lambda *a, **k: None
sys.modules['pyranges'] = pyranges_mock

import cptac

cancer_type = sys.argv[1] if len(sys.argv) > 1 else "ucec"
cptac_class = {"ucec": "Ucec", "coad": "Coad", "luad": "Luad"}[cancer_type]

print(f"Loading CPTAC {cancer_type.upper()}...")
dataset = getattr(cptac, cptac_class)()

try:
    prot_raw = dataset.get_proteomics(source='umich')
except Exception:
    try:
        prot_raw = dataset.get_proteomics(source='harmonized')
    except Exception:
        prot_raw = dataset.get_proteomics()

# Flatten MultiIndex columns to gene symbols
if isinstance(prot_raw.columns, pd.MultiIndex):
    prot_df = prot_raw.copy()
    prot_df.columns = prot_raw.columns.get_level_values(0)
else:
    prot_df = prot_raw.copy()

# Average duplicate gene columns
prot_df = prot_df.T.groupby(level=0).mean().T

# Sample type
sample_type = pd.Series(
    ["Normal" if str(s).endswith(".N") or "Normal" in str(s) else "Tumor"
     for s in prot_df.index],
    index=prot_df.index
)

# Filter: <50% missing
missing_frac = prot_df.isnull().mean(axis=0)
prot_df = prot_df.loc[:, missing_frac < 0.5]

print(f"  Matrix: {prot_df.shape[0]} samples x {prot_df.shape[1]} proteins")
print(f"  Tumor: {(sample_type == 'Tumor').sum()}, Normal: {(sample_type == 'Normal').sum()}")

# Save
out_dir = Path(f"data/{cancer_type}/raw")
out_dir.mkdir(parents=True, exist_ok=True)

prot_df.to_csv(out_dir / "expr_matrix.csv")
sample_type.to_frame("group").to_csv(out_dir / "sample_labels.csv")

print(f"  Saved: {out_dir}/expr_matrix.csv, sample_labels.csv")
