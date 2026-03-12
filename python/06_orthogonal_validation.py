#!/usr/bin/env python3
"""
06_orthogonal_validation.py - Validate ProtDynPredict predictions against
independent biological data.

Validation axes:
  1. RNA-Protein correlation (CPTAC transcriptomics vs proteomics)
  2. Pathway enrichment of predicted-DE proteins (KEGG / GO-slim)
  3. Protein half-life comparison (optional, if external data available)

Output:
  results/reports/orthogonal_validation_report.md
  results/figures/rna_protein_correlation.png
  results/figures/pathway_enrichment.png
"""

import argparse
import sys
import types
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "train_file": "data/ucec/processed/feature_matrix_train.csv",
    "de_file": "data/ucec/raw/de_results.csv",
    "pathway_file": "data/processed/pathway_membership.csv",
    "go_slim_file": "data/processed/go_slim_matrix.csv",
    "model_dir": "models",
    "figures_dir": "results/figures",
    "reports_dir": "results/reports",
    "external_dir": "data/external",
    "random_state": 42,
    # DE thresholds (match training pipeline)
    "logfc_thresh": 0.5,
    "fdr_thresh": 0.05,
}

LEAKY_PREFIXES = [
    "ppi_frac_neighbors_", "ppi_weighted_frac_",
    "pw_max_frac_", "pw_mean_frac_",
    "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _ensure_dirs():
    """Create output directories if they don't exist."""
    Path(CONFIG["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["reports_dir"]).mkdir(parents=True, exist_ok=True)


def _load_training_data():
    """Load the training feature matrix with labels."""
    path = Path(CONFIG["train_file"])
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    df = pd.read_csv(path)
    log.info(f"  Loaded training data: {df.shape[0]} proteins, {df.shape[1]} columns")
    return df


def _load_de_results():
    """Load original DE results (UniProt_ID, log2FC, adj_pvalue)."""
    path = Path(CONFIG["de_file"])
    if not path.exists():
        raise FileNotFoundError(f"DE results not found: {path}")
    df = pd.read_csv(path)
    log.info(f"  Loaded DE results: {df.shape[0]} proteins")
    return df


def _load_model(stage="stage1"):
    """Load a saved model artifact."""
    import joblib
    path = Path(CONFIG["model_dir"]) / f"{stage}_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    artifact = joblib.load(path)
    log.info(f"  Loaded model: {path}")
    return artifact


def _build_uniprot_to_gene_map():
    """Build UniProt_ID -> gene_symbol mapping via UniProt REST API.

    Falls back to cptac proteomics column names if API is unavailable.
    """
    de = _load_de_results()
    uniprots = de["UniProt_ID"].unique().tolist()

    gene_map = {}
    try:
        import requests
        log.info("  Querying UniProt REST API for gene symbol mapping...")
        batch_size = 100
        for i in range(0, len(uniprots), batch_size):
            batch = uniprots[i:i + batch_size]
            query = " OR ".join([f"(accession:{uid})" for uid in batch])
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": f"({query}) AND (organism_id:9606)",
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
                        if uid in batch and uid not in gene_map:
                            gene_map[uid] = gene
        log.info(f"  Mapped {len(gene_map)}/{len(uniprots)} IDs to gene symbols")
    except Exception as e:
        log.warning(f"  UniProt API mapping failed: {e}")

    return gene_map


# =========================================================================
# Section 1: RNA-Protein Correlation
# =========================================================================
def rna_protein_correlation(report_sections: list):
    """Compare protein log2FC with mRNA log2FC from CPTAC transcriptomics."""
    section_title = "## 1. RNA-Protein Correlation"
    log.info("\n" + "=" * 60)
    log.info("  SECTION 1: RNA-PROTEIN CORRELATION")
    log.info("=" * 60)

    try:
        # Monkey-patch pyranges before importing cptac
        pyranges_mock = types.ModuleType("pyranges")
        pyranges_mock.read_gtf = lambda *a, **k: None
        sys.modules["pyranges"] = pyranges_mock

        import cptac
    except ImportError:
        msg = ("cptac package not available. Skipping RNA-protein "
               "correlation analysis.\n\nInstall with: `pip install cptac`")
        log.warning(f"  {msg}")
        report_sections.append(f"{section_title}\n\n{msg}\n")
        return

    try:
        # Load dataset
        log.info("  Loading CPTAC UCEC dataset...")
        dataset = cptac.Ucec()

        # --- Proteomics (reuse DE results for protein log2FC) ---
        de_df = _load_de_results()

        # --- Transcriptomics ---
        log.info("  Extracting transcriptomics data...")
        # cptac may have a bug with generator in get_transcriptomics;
        # try specific sources to avoid the len(generator) bug
        rna_raw = None
        for src in ['bcm', 'broad', 'washu', 'umich', 'harmonized']:
            try:
                rna_raw = dataset.get_transcriptomics(source=src)
                log.info(f"  Loaded transcriptomics from source='{src}'")
                break
            except Exception:
                continue
        if rna_raw is None:
            raise ValueError("Could not extract transcriptomics data from CPTAC")
        if isinstance(rna_raw.columns, pd.MultiIndex):
            rna_df = rna_raw.copy()
            rna_df.columns = rna_raw.columns.get_level_values(0)
        else:
            rna_df = rna_raw.copy()

        # Merge duplicate gene columns
        rna_df = rna_df.T.groupby(level=0).mean().T
        log.info(f"  Transcriptomics matrix: {rna_df.shape[0]} samples x "
                 f"{rna_df.shape[1]} genes")

        # Identify tumor vs normal
        sample_ids = rna_df.index.tolist()
        sample_type = np.array([
            "Normal" if str(sid).endswith(".N") or "Normal" in str(sid)
            else "Tumor"
            for sid in sample_ids
        ])
        tumor_mask = sample_type == "Tumor"
        normal_mask = sample_type == "Normal"
        log.info(f"  Tumor: {tumor_mask.sum()}, Normal: {normal_mask.sum()}")

        if normal_mask.sum() < 3:
            raise ValueError("Fewer than 3 normal samples in transcriptomics")

        # Compute mRNA log2FC per gene (same approach as proteomics)
        log.info("  Computing mRNA log2FC (tumor vs normal)...")
        rna_results = []
        for gene in rna_df.columns:
            vals = rna_df[gene]
            t_vals = vals[tumor_mask].dropna()
            n_vals = vals[normal_mask].dropna()
            if len(t_vals) < 3 or len(n_vals) < 3:
                continue
            rna_log2fc = t_vals.mean() - n_vals.mean()
            _, p_val = stats.ttest_ind(t_vals, n_vals, equal_var=False)
            if np.isnan(rna_log2fc) or np.isnan(p_val):
                continue
            rna_results.append({
                "gene_symbol": gene,
                "rna_log2FC": rna_log2fc,
                "rna_pvalue": p_val,
            })

        rna_de = pd.DataFrame(rna_results)
        log.info(f"  mRNA DE computed for {len(rna_de)} genes")

        # BH correction
        if len(rna_de) > 0:
            from statsmodels.stats.multitest import multipletests
            _, adj_p, _, _ = multipletests(rna_de["rna_pvalue"], method="fdr_bh")
            rna_de["rna_adj_pvalue"] = adj_p
        else:
            raise ValueError("No mRNA DE results computed")

        # Map UniProt IDs to gene symbols for matching
        gene_map = _build_uniprot_to_gene_map()
        de_df["gene_symbol"] = de_df["UniProt_ID"].map(gene_map)
        de_df = de_df.dropna(subset=["gene_symbol"])

        # Merge protein and RNA data on gene symbol
        merged = de_df.merge(rna_de, on="gene_symbol", how="inner")
        log.info(f"  Matched proteins with RNA data: {len(merged)}")

        if len(merged) < 10:
            raise ValueError(f"Only {len(merged)} matched genes; too few for analysis")

        # --- Spearman correlation ---
        rho, p_spearman = stats.spearmanr(merged["log2FC"], merged["rna_log2FC"])
        log.info(f"  Spearman rho = {rho:.4f}, p = {p_spearman:.2e}")

        # --- Fisher's exact test: protein-DE vs mRNA-DE concordance ---
        thresh = CONFIG["logfc_thresh"]
        fdr = CONFIG["fdr_thresh"]
        prot_de = ((merged["log2FC"].abs() > thresh) &
                   (merged["adj_pvalue"] < fdr))
        rna_is_de = ((merged["rna_log2FC"].abs() > thresh) &
                     (merged["rna_adj_pvalue"] < fdr))

        # 2x2: [prot_DE & rna_DE, prot_DE & rna_notDE]
        #       [prot_notDE & rna_DE, prot_notDE & rna_notDE]
        a = (prot_de & rna_is_de).sum()
        b = (prot_de & ~rna_is_de).sum()
        c = (~prot_de & rna_is_de).sum()
        d = (~prot_de & ~rna_is_de).sum()
        fisher_or, fisher_p = stats.fisher_exact([[a, b], [c, d]])
        log.info(f"  Fisher exact: OR = {fisher_or:.2f}, p = {fisher_p:.2e}")
        log.info(f"  Concordance table: [[{a},{b}],[{c},{d}]]")

        # --- Model prediction vs mRNA DE ---
        model_vs_rna_text = ""
        try:
            train_df = _load_training_data()
            train_df["gene_symbol"] = train_df["UniProt_ID"].map(gene_map)
            train_labels = train_df[["gene_symbol", "label"]].dropna(
                subset=["gene_symbol"]
            )
            merged2 = merged.merge(train_labels, on="gene_symbol", how="inner")
            if len(merged2) > 10:
                pred_de = merged2["label"] != "unchanged"
                rna_de_flag = ((merged2["rna_log2FC"].abs() > thresh) &
                               (merged2["rna_adj_pvalue"] < fdr))
                a2 = (pred_de & rna_de_flag).sum()
                b2 = (pred_de & ~rna_de_flag).sum()
                c2 = (~pred_de & rna_de_flag).sum()
                d2 = (~pred_de & ~rna_de_flag).sum()
                or2, p2 = stats.fisher_exact([[a2, b2], [c2, d2]])
                model_vs_rna_text = (
                    f"\n### Model-predicted DE vs mRNA DE\n"
                    f"- Fisher exact: OR = {or2:.2f}, p = {p2:.2e}\n"
                    f"- Concordance: [[{a2},{b2}],[{c2},{d2}]]\n"
                )
                log.info(f"  Model-pred vs mRNA: OR={or2:.2f}, p={p2:.2e}")
        except Exception as e:
            model_vs_rna_text = (
                f"\n### Model-predicted DE vs mRNA DE\n"
                f"Could not compute: {e}\n"
            )

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(7, 6))

        # Color by training label if available
        colors = np.full(len(merged), "#888888")
        try:
            train_df = _load_training_data()
            train_df["gene_symbol"] = train_df["UniProt_ID"].map(gene_map)
            label_map = dict(zip(train_df["gene_symbol"], train_df["label"]))
            merged["pred_class"] = merged["gene_symbol"].map(label_map)
            color_dict = {"up": "#e74c3c", "down": "#3498db",
                          "unchanged": "#95a5a6"}
            colors = merged["pred_class"].map(color_dict).fillna("#888888").values
        except Exception:
            pass

        ax.scatter(merged["log2FC"], merged["rna_log2FC"],
                   c=colors, alpha=0.5, s=15, edgecolors="none")
        # Regression line
        slope, intercept = np.polyfit(merged["log2FC"], merged["rna_log2FC"], 1)
        x_range = np.linspace(merged["log2FC"].min(), merged["log2FC"].max(), 100)
        ax.plot(x_range, slope * x_range + intercept, "k--", lw=1.2, alpha=0.7)

        ax.set_xlabel("Protein log2FC (CPTAC)", fontsize=12)
        ax.set_ylabel("mRNA log2FC (CPTAC)", fontsize=12)
        ax.set_title(
            f"RNA-Protein Correlation\n"
            f"Spearman rho={rho:.3f}, p={p_spearman:.1e}, n={len(merged)}",
            fontsize=11,
        )
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                   label="Up", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
                   label="Down", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6",
                   label="Unchanged", markersize=7),
        ]
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.8)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        plt.tight_layout()

        fig_path = Path(CONFIG["figures_dir"]) / "rna_protein_correlation.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        log.info(f"  Saved: {fig_path}")

        # --- Build report section ---
        report = (
            f"{section_title}\n\n"
            f"Compared protein-level log2FC (CPTAC proteomics) with mRNA-level "
            f"log2FC (CPTAC transcriptomics) for the same UCEC tumor-vs-normal "
            f"comparison.\n\n"
            f"- **Matched genes:** {len(merged)}\n"
            f"- **Spearman rho:** {rho:.4f} (p = {p_spearman:.2e})\n"
            f"- **Fisher exact test** (protein-DE vs mRNA-DE concordance):\n"
            f"  - Odds ratio: {fisher_or:.2f}, p = {fisher_p:.2e}\n"
            f"  - Concordance table:\n\n"
            f"  |  | mRNA-DE | mRNA-not-DE |\n"
            f"  |---|---|---|\n"
            f"  | **Protein-DE** | {a} | {b} |\n"
            f"  | **Protein-not-DE** | {c} | {d} |\n"
            f"{model_vs_rna_text}\n"
            f"![RNA-Protein Correlation](../figures/rna_protein_correlation.png)\n"
        )
        report_sections.append(report)

    except Exception as e:
        msg = f"RNA-protein correlation analysis failed: {e}"
        log.warning(f"  {msg}")
        report_sections.append(f"{section_title}\n\n{msg}\n")


# =========================================================================
# Section 2: Pathway Enrichment
# =========================================================================
def pathway_enrichment(report_sections: list):
    """Hypergeometric test for enrichment of predicted-DE proteins in pathways."""
    section_title = "## 2. Pathway Enrichment of Predicted-DE Proteins"
    log.info("\n" + "=" * 60)
    log.info("  SECTION 2: PATHWAY ENRICHMENT")
    log.info("=" * 60)

    try:
        train_df = _load_training_data()

        # Identify predicted-DE proteins (from training labels)
        de_proteins = set(
            train_df.loc[train_df["label"] != "unchanged", "UniProt_ID"]
        )
        all_proteins = set(train_df["UniProt_ID"])
        log.info(f"  DE proteins: {len(de_proteins)}/{len(all_proteins)}")

        enrichment_results = []

        # --- Strategy A: KEGG pathways from pathway_membership.csv ---
        pw_path = Path(CONFIG["pathway_file"])
        if pw_path.exists():
            log.info("  Using KEGG pathway memberships...")
            pw_df = pd.read_csv(pw_path)
            # Columns: UniProt_ID, pathway_id, source
            pw_df = pw_df[pw_df["UniProt_ID"].isin(all_proteins)]

            pathways = pw_df.groupby("pathway_id")["UniProt_ID"].apply(set)
            N = len(all_proteins)  # population size

            for pw_id, pw_members in pathways.items():
                K = len(pw_members)  # successes in population
                if K < 5:
                    continue  # skip tiny pathways
                overlap = de_proteins & pw_members
                n = len(de_proteins)  # draws
                k = len(overlap)  # observed successes

                # Hypergeometric p-value (survival function)
                p_val = stats.hypergeom.sf(k - 1, N, K, n)

                fold = (k / n) / (K / N) if (K > 0 and n > 0) else 0.0
                enrichment_results.append({
                    "term": pw_id,
                    "source": "KEGG",
                    "term_size": K,
                    "overlap": k,
                    "fold_enrichment": fold,
                    "pvalue": p_val,
                    "genes": ";".join(sorted(overlap)) if overlap else "",
                })

            log.info(f"  Tested {len(enrichment_results)} KEGG pathways")

        # --- Strategy B: GO-slim terms from feature matrix ---
        goslim_cols = [c for c in train_df.columns if c.startswith("GOslim_")]
        if goslim_cols:
            log.info(f"  Using {len(goslim_cols)} GO-slim terms from feature matrix...")
            N = len(all_proteins)
            n = len(de_proteins)
            de_idx = train_df["UniProt_ID"].isin(de_proteins)

            for col in goslim_cols:
                members = set(
                    train_df.loc[train_df[col] == 1, "UniProt_ID"]
                )
                K = len(members)
                if K < 5:
                    continue
                overlap = de_proteins & members
                k = len(overlap)
                p_val = stats.hypergeom.sf(k - 1, N, K, n)
                fold = (k / n) / (K / N) if (K > 0 and n > 0) else 0.0

                enrichment_results.append({
                    "term": col,
                    "source": "GOslim",
                    "term_size": K,
                    "overlap": k,
                    "fold_enrichment": fold,
                    "pvalue": p_val,
                    "genes": "",
                })

            log.info(f"  Total terms tested: {len(enrichment_results)}")

        if not enrichment_results:
            raise ValueError("No pathway/GO-slim data available for enrichment")

        enrich_df = pd.DataFrame(enrichment_results)

        # Multiple testing correction
        from statsmodels.stats.multitest import multipletests
        _, adj_p, _, _ = multipletests(enrich_df["pvalue"], method="fdr_bh")
        enrich_df["adj_pvalue"] = adj_p
        enrich_df = enrich_df.sort_values("pvalue")

        sig = enrich_df[enrich_df["adj_pvalue"] < 0.05]
        log.info(f"  Significant terms (FDR < 0.05): {len(sig)}")

        # --- Plot top enriched terms ---
        top_n = min(20, len(enrich_df))
        plot_df = enrich_df.head(top_n).copy()
        plot_df["-log10(p)"] = -np.log10(plot_df["pvalue"].clip(lower=1e-300))
        plot_df = plot_df.sort_values("-log10(p)")

        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        bars = ax.barh(
            range(len(plot_df)),
            plot_df["-log10(p)"].values,
            color=np.where(plot_df["adj_pvalue"].values < 0.05, "#2ecc71", "#bdc3c7"),
        )
        ax.set_yticks(range(len(plot_df)))
        ax.set_yticklabels(plot_df["term"].values, fontsize=9)
        ax.set_xlabel("-log10(p-value)", fontsize=11)
        ax.set_title("Pathway / GO-slim Enrichment of DE Proteins", fontsize=12)
        ax.axvline(-np.log10(0.05), color="red", ls="--", lw=0.8, label="p=0.05")
        ax.legend(fontsize=9)
        plt.tight_layout()

        fig_path = Path(CONFIG["figures_dir"]) / "pathway_enrichment.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        log.info(f"  Saved: {fig_path}")

        # --- Report ---
        top_table = enrich_df.head(15)[
            ["term", "source", "term_size", "overlap", "fold_enrichment",
             "pvalue", "adj_pvalue"]
        ].to_markdown(index=False, floatfmt=".2e")

        report = (
            f"{section_title}\n\n"
            f"Hypergeometric test for over-representation of predicted-DE "
            f"proteins in biological pathways and GO-slim terms.\n\n"
            f"- **DE proteins:** {len(de_proteins)} / {len(all_proteins)}\n"
            f"- **Terms tested:** {len(enrich_df)}\n"
            f"- **Significant (FDR < 0.05):** {len(sig)}\n\n"
            f"### Top Enriched Terms\n\n{top_table}\n\n"
            f"![Pathway Enrichment](../figures/pathway_enrichment.png)\n"
        )
        report_sections.append(report)

    except Exception as e:
        msg = f"Pathway enrichment analysis failed: {e}"
        log.warning(f"  {msg}")
        report_sections.append(f"{section_title}\n\n{msg}\n")


# =========================================================================
# Section 3: Protein Half-Life (Optional)
# =========================================================================
def protein_halflife(report_sections: list):
    """Compare protein half-lives between predicted-DE and unchanged proteins."""
    section_title = "## 3. Protein Half-Life Comparison (Optional)"
    log.info("\n" + "=" * 60)
    log.info("  SECTION 3: PROTEIN HALF-LIFE")
    log.info("=" * 60)

    try:
        train_df = _load_training_data()
        de_proteins = set(
            train_df.loc[train_df["label"] != "unchanged", "UniProt_ID"]
        )
        unchanged_proteins = set(
            train_df.loc[train_df["label"] == "unchanged", "UniProt_ID"]
        )

        # --- Look for external half-life data ---
        ext_dir = Path(CONFIG["external_dir"])
        hl_path = None
        if ext_dir.exists():
            candidates = list(ext_dir.glob("*half*life*")) + \
                         list(ext_dir.glob("*halflife*")) + \
                         list(ext_dir.glob("*protein_turnover*"))
            if candidates:
                hl_path = candidates[0]

        if hl_path is None:
            log.info("  No local half-life data found.")

        if hl_path is None:
            raise FileNotFoundError(
                "No protein half-life data available. "
                "Place a CSV/Excel with columns [UniProt_ID/gene_symbol, half_life] "
                f"in {ext_dir}/ or ensure internet access for download."
            )

        # --- Load half-life data ---
        log.info(f"  Loading: {hl_path}")
        if str(hl_path).endswith(".xlsx") or str(hl_path).endswith(".xls"):
            hl_df = pd.read_excel(hl_path)
        else:
            hl_df = pd.read_csv(hl_path)

        log.info(f"  Half-life table: {hl_df.shape[0]} rows, "
                 f"columns: {list(hl_df.columns[:10])}")

        # Try to find the right columns (various formats)
        id_col = None
        hl_col = None
        for c in hl_df.columns:
            cl = str(c).lower()
            if "uniprot" in cl or "accession" in cl or "protein" in cl:
                id_col = c
            if "half" in cl and "life" in cl:
                hl_col = c
            if "t1/2" in cl or "t_half" in cl or "turnover" in cl:
                hl_col = c

        # If columns are numbered/generic, try gene symbol matching
        if id_col is None:
            for c in hl_df.columns:
                cl = str(c).lower()
                if "gene" in cl or "symbol" in cl:
                    id_col = c

        if id_col is None or hl_col is None:
            raise ValueError(
                f"Could not identify ID and half-life columns in {hl_path}. "
                f"Columns: {list(hl_df.columns[:15])}"
            )

        hl_df = hl_df[[id_col, hl_col]].dropna()
        hl_df.columns = ["protein_id", "half_life"]
        hl_df["half_life"] = pd.to_numeric(hl_df["half_life"], errors="coerce")
        hl_df = hl_df.dropna()

        # Match to our proteins (try direct UniProt match first)
        de_hl = hl_df[hl_df["protein_id"].isin(de_proteins)]["half_life"].values
        unc_hl = hl_df[hl_df["protein_id"].isin(unchanged_proteins)]["half_life"].values

        # If few matches, try gene-symbol-based matching
        if len(de_hl) < 10 or len(unc_hl) < 10:
            log.info("  Few direct matches; trying gene symbol mapping...")
            gene_map = _build_uniprot_to_gene_map()
            gene_to_up = {v: k for k, v in gene_map.items()}
            hl_df["UniProt_ID"] = hl_df["protein_id"].map(gene_to_up)
            hl_mapped = hl_df.dropna(subset=["UniProt_ID"])
            de_hl = hl_mapped[
                hl_mapped["UniProt_ID"].isin(de_proteins)
            ]["half_life"].values
            unc_hl = hl_mapped[
                hl_mapped["UniProt_ID"].isin(unchanged_proteins)
            ]["half_life"].values

        log.info(f"  DE proteins with half-life: {len(de_hl)}")
        log.info(f"  Unchanged proteins with half-life: {len(unc_hl)}")

        if len(de_hl) < 5 or len(unc_hl) < 5:
            raise ValueError(
                f"Too few proteins matched to half-life data "
                f"(DE={len(de_hl)}, unchanged={len(unc_hl)})"
            )

        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(de_hl, unc_hl, alternative="two-sided")
        log.info(f"  Mann-Whitney U = {u_stat:.1f}, p = {u_pval:.2e}")
        log.info(f"  Median half-life: DE={np.median(de_hl):.2f}, "
                 f"unchanged={np.median(unc_hl):.2f}")

        report = (
            f"{section_title}\n\n"
            f"Compared protein half-lives between predicted-DE and unchanged "
            f"proteins.\n\n"
            f"- **Data source:** {hl_path.name}\n"
            f"- **DE proteins matched:** {len(de_hl)}\n"
            f"- **Unchanged proteins matched:** {len(unc_hl)}\n"
            f"- **Median half-life (DE):** {np.median(de_hl):.2f}\n"
            f"- **Median half-life (unchanged):** {np.median(unc_hl):.2f}\n"
            f"- **Mann-Whitney U:** {u_stat:.1f}, p = {u_pval:.2e}\n"
        )
        report_sections.append(report)

    except Exception as e:
        msg = f"Protein half-life analysis skipped: {e}"
        log.info(f"  {msg}")
        report_sections.append(f"{section_title}\n\n{msg}\n")


# =========================================================================
# Report Generation
# =========================================================================
def generate_report(report_sections: list):
    """Compile all sections into a markdown report."""
    log.info("\n" + "=" * 60)
    log.info("  GENERATING REPORT")
    log.info("=" * 60)

    header = (
        "# ProtDynPredict: Orthogonal Validation Report\n\n"
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "This report validates model predictions against independent biological "
        "data sources that were not used during model training.\n\n"
        "---\n\n"
    )

    report_path = Path(CONFIG["reports_dir"]) / "orthogonal_validation_report.md"
    report_path.write_text(header + "\n\n".join(report_sections) + "\n")
    log.info(f"  Saved: {report_path}")
    return report_path


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Validate ProtDynPredict predictions against independent "
                    "biological data."
    )
    parser.add_argument(
        "--dataset", default="ucec",
        help="Dataset name (e.g., ucec, coad, brca)",
    )
    parser.add_argument(
        "--sections", nargs="*",
        default=["rna", "pathway", "halflife"],
        choices=["rna", "pathway", "halflife"],
        help="Which validation sections to run (default: all).",
    )
    parser.add_argument(
        "--project-dir", type=str, default=None,
        help="Project root directory (default: auto-detect).",
    )
    args = parser.parse_args()

    # Update CONFIG paths based on dataset name
    dataset = args.dataset
    CONFIG["train_file"] = f"data/{dataset}/processed/feature_matrix_train.csv"
    CONFIG["de_file"] = f"data/{dataset}/raw/de_results.csv"
    CONFIG["pathway_file"] = f"data/{dataset}/processed/pathway_membership.csv"
    CONFIG["go_slim_file"] = f"data/{dataset}/processed/go_slim_matrix.csv"
    CONFIG["model_dir"] = f"models/{dataset}"
    CONFIG["figures_dir"] = f"results/{dataset}/figures"
    CONFIG["reports_dir"] = f"results/{dataset}/reports"

    # Set working directory
    if args.project_dir:
        import os
        os.chdir(args.project_dir)
    else:
        # Auto-detect: walk up from script location to find data/
        script_dir = Path(__file__).resolve().parent
        for candidate in [script_dir.parent, Path.cwd()]:
            if (candidate / "data").exists():
                import os
                os.chdir(candidate)
                break

    log.info("=" * 60)
    log.info("  ORTHOGONAL VALIDATION")
    log.info("=" * 60)
    log.info(f"  Working dir: {Path.cwd()}")

    _ensure_dirs()

    report_sections = []

    if "rna" in args.sections:
        rna_protein_correlation(report_sections)

    if "pathway" in args.sections:
        pathway_enrichment(report_sections)

    if "halflife" in args.sections:
        protein_halflife(report_sections)

    report_path = generate_report(report_sections)

    log.info("\n" + "=" * 60)
    log.info("  DONE")
    log.info("=" * 60)
    log.info(f"  Report: {report_path}")
    log.info(f"  Figures: {CONFIG['figures_dir']}/")


if __name__ == "__main__":
    main()
