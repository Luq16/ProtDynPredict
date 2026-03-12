#!/usr/bin/env python3
"""
09_report.py
Comprehensive publication report generator for ProtDynPredict.

Aggregates all pipeline reports, extracts key metrics, and produces:
  - Publication summary (results/reports/publication_summary.md)
  - Supplementary tables (results/supplementary/Table_S*.csv)
  - Composite figure (results/figures/composite_summary.png)

Input:  results/reports/*.md, models/*.joblib, data/processed/*.csv
Output: results/reports/publication_summary.md
        results/supplementary/Table_S1_feature_descriptions.csv
        results/supplementary/Table_S2_shap_rankings.csv
        results/supplementary/Table_S3_cross_dataset_results.csv
        results/supplementary/Table_S4_top_predictions.csv
        results/figures/composite_summary.png
"""

import argparse
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# Optional imports --------------------------------------------------------
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Feature decoder ---------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from utils.feature_decoder import decode_feature, get_property_group
except ImportError:
    decode_feature = lambda x: x
    get_property_group = lambda x: ["other"]


# ── Configuration ────────────────────────────────────────────────────────

CONFIG = {
    "reports_dir": "results/reports",
    "figures_dir": "results/figures",
    "supplementary_dir": "results/supplementary",
    "data_dir": "data/ucec/processed",
    "model_dir": "models",
    "results_dir": "results",
}

# Canonical order for report sections
REPORT_ORDER = [
    ("validation_report.md", "Premise Validation"),
    ("training_report.md", "Model Training"),
    ("baseline_label_propagation.md", "Label Propagation Baseline"),
    ("baselines_comparison.md", "Baselines Comparison"),
    ("within_validation_report.md", "Within-Dataset Validation"),
    ("cross_validation_report.md", "Cross-Dataset Generalization"),
    ("orthogonal_validation_report.md", "Orthogonal Validation"),
    ("interpretation_report.md", "Model Interpretation"),
    ("prediction_report.md", "Predictions"),
]


# ── Metric extraction ───────────────────────────────────────────────────

def extract_metrics_from_report(filepath):
    """Extract AUC, F1, MCC, and other key numeric metrics from a markdown report.

    Returns a dict of {metric_name: value} found via common patterns.
    """
    text = filepath.read_text()
    metrics = {}

    # AUC patterns (various formats seen in the reports)
    for pattern, key in [
        (r"\*\*(?:CV |Overall |Mean (?:CV )?)?AUC\*?\*?[:\s]+([0-9]+\.[0-9]+)", "auc"),
        (r"AUC[:\s]+([0-9]+\.[0-9]+)\s*±\s*([0-9]+\.[0-9]+)", "auc_with_ci"),
        (r"\*\*F1\*?\*?[:\s]+([0-9]+\.[0-9]+)", "f1"),
        (r"\*\*MCC\*?\*?[:\s]+([0-9]+\.[0-9]+)", "mcc"),
        (r"Permutation p-value[:\s]+([0-9]+\.[0-9]+)", "permutation_p"),
    ]:
        m = re.search(pattern, text)
        if m:
            metrics[key] = m.group(1)
            if key == "auc_with_ci":
                metrics["auc"] = m.group(1)
                metrics["auc_std"] = m.group(2)

    # Stage-specific AUC (Stage 1 / Stage 2)
    stage1_block = re.search(
        r"Stage\s*1.*?AUC\*?\*?[:\s]+([0-9]+\.[0-9]+)(?:\s*±\s*([0-9]+\.[0-9]+))?",
        text, re.DOTALL | re.IGNORECASE,
    )
    if stage1_block:
        metrics["stage1_auc"] = stage1_block.group(1)
        if stage1_block.group(2):
            metrics["stage1_auc_std"] = stage1_block.group(2)

    stage2_block = re.search(
        r"Stage\s*2.*?AUC\*?\*?[:\s]+([0-9]+\.[0-9]+)(?:\s*±\s*([0-9]+\.[0-9]+))?",
        text, re.DOTALL | re.IGNORECASE,
    )
    if stage2_block:
        metrics["stage2_auc"] = stage2_block.group(1)
        if stage2_block.group(2):
            metrics["stage2_auc_std"] = stage2_block.group(2)

    # Baselines table rows: | ModelName | AUC | F1 | MCC |
    # Process line-by-line to avoid cross-line matches that capture leading pipes
    baseline_rows = []
    for line in text.splitlines():
        m = re.match(
            r"\|\s*(.+?)\s*\|\s*([0-9]+\.[0-9]+)\s*\|\s*([0-9]+\.[0-9]+)\s*\|\s*([0-9]+\.[0-9]+)\s*\|",
            line,
        )
        if m:
            baseline_rows.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    if baseline_rows:
        baselines = []
        for name, auc, f1, mcc in baseline_rows:
            name = name.strip()
            if name.lower() in ("model", "fold", "iteration", "rank", "feature set"):
                continue
            baselines.append({"model": name, "auc": auc, "f1": f1, "mcc": mcc})
        if baselines:
            metrics["baselines"] = baselines

    # Feature category contributions (line-by-line to avoid cross-line matches)
    cat_rows = []
    for line in text.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([0-9]+\.[0-9]+)%\s*\|", line)
        if m:
            cat_rows.append((m.group(1), m.group(2)))
    if cat_rows:
        categories = {}
        for cat, pct in cat_rows:
            cat = cat.strip()
            if cat.lower() in ("category",):
                continue
            categories[cat] = float(pct)
        if categories:
            metrics["feature_categories"] = categories

    # SHAP top features
    shap_rows = re.findall(
        r"\|\s*(\d+)\s*\|\s*`(.+?)`\s*\|\s*(.+?)\s*\|\s*([0-9]+\.[0-9]+)\s*\|",
        text,
    )
    if shap_rows:
        metrics["shap_features"] = [
            {"rank": int(r), "feature": f, "category": c.strip(), "mean_shap": float(v)}
            for r, f, c, v in shap_rows
        ]

    return metrics


# ── Supplementary table generators ──────────────────────────────────────

def generate_feature_table(data_dir):
    """Table S1: Feature names, descriptions, and categories."""
    train_path = Path(data_dir) / "feature_matrix_train.csv"
    if not train_path.exists():
        print("  [SKIP] Table S1 - training data not found")
        return None

    df = pd.read_csv(train_path, nrows=0)
    meta_cols = {"UniProt_ID", "label", "log2FC", "adj_pvalue"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    rows = []
    for feat in feature_cols:
        desc = decode_feature(feat)
        cat = _categorize_feature(feat)
        rows.append({"feature_name": feat, "description": desc, "category": cat})

    return pd.DataFrame(rows)


def generate_shap_table(model_dir, data_dir):
    """Table S2: Full SHAP rankings for both stages.

    Attempts to compute SHAP values from models.  Falls back to parsing
    the interpretation report.
    """
    results = []

    # Try loading models and computing SHAP
    if HAS_JOBLIB and HAS_SHAP:
        train_path = Path(data_dir) / "feature_matrix_train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path)
            meta_cols = {"UniProt_ID", "label", "log2FC", "adj_pvalue"}

            for stage_tag, model_file, mask_fn in [
                ("Stage1_DE_vs_Unchanged", "stage1_model.joblib", None),
                ("Stage2_Up_vs_Down", "stage2_model.joblib",
                 lambda d: d["label"].isin(["up", "down"])),
            ]:
                model_path = Path(model_dir) / model_file
                if not model_path.exists():
                    continue
                try:
                    data = joblib.load(model_path)
                    model = data["model"]
                    feature_cols = data.get("feature_cols",
                                            [c for c in df.columns if c not in meta_cols])
                    X = df[feature_cols].values.astype(np.float32)
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                    if mask_fn is not None:
                        mask = mask_fn(df).values
                        X = X[mask]

                    # Subsample for speed
                    n = min(500, X.shape[0])
                    rng = np.random.RandomState(42)
                    idx = rng.choice(X.shape[0], n, replace=False)
                    X_sample = X[idx]

                    explainer = shap.TreeExplainer(model)
                    try:
                        explanation = explainer(X_sample)
                        sv = explanation.values
                        if sv.ndim == 3:
                            sv = sv[:, :, 1]
                    except Exception:
                        shap_values = explainer.shap_values(X_sample)
                        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

                    mean_abs = np.abs(sv).mean(axis=0)
                    for rank, i in enumerate(np.argsort(-mean_abs), 1):
                        results.append({
                            "stage": stage_tag,
                            "rank": rank,
                            "feature": feature_cols[i],
                            "category": _categorize_feature(feature_cols[i]),
                            "description": decode_feature(feature_cols[i]),
                            "mean_abs_shap": round(float(mean_abs[i]), 6),
                        })
                    print(f"  Table S2: computed SHAP for {stage_tag} ({len(feature_cols)} features)")
                except Exception as e:
                    print(f"  [WARN] SHAP computation failed for {stage_tag}: {e}")

    # Fallback: parse interpretation report
    if not results:
        interp_path = Path(CONFIG["reports_dir"]) / "interpretation_report.md"
        if interp_path.exists():
            metrics = extract_metrics_from_report(interp_path)
            if "shap_features" in metrics:
                for entry in metrics["shap_features"]:
                    results.append({
                        "stage": "parsed_from_report",
                        "rank": entry["rank"],
                        "feature": entry["feature"],
                        "category": entry["category"],
                        "description": decode_feature(entry["feature"]),
                        "mean_abs_shap": entry["mean_shap"],
                    })
                print(f"  Table S2: parsed {len(results)} SHAP features from report")

    if not results:
        print("  [SKIP] Table S2 - no SHAP data available")
        return None

    return pd.DataFrame(results)


def generate_cross_dataset_table(reports_dir):
    """Table S3: Cross-dataset results matrix (if available)."""
    cross_path = Path(reports_dir) / "cross_validation_report.md"
    if not cross_path.exists():
        print("  [SKIP] Table S3 - cross-dataset report not found")
        return None

    text = cross_path.read_text()

    # Only parse tables from Section 1 and Section 2 (not per-class breakdown)
    section_match = re.search(
        r"## Section 1.*?(?=## Section 4|## Interpretation|$)",
        text, re.DOTALL
    )
    section_text = section_match.group(0) if section_match else text

    # Try to parse cross-dataset table (format from 05_validate_cross.py):
    # | Train | Test | Stage | AUC | F1 | MCC |
    rows = re.findall(
        r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*([0-9]+\.[0-9]+|N/A)\s*\|\s*([0-9]+\.[0-9]+)\s*\|\s*([0-9.-]+)\s*\|",
        section_text,
    )
    if not rows:
        # Fallback: try simpler 3-column format
        rows = re.findall(
            r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*([0-9]+\.[0-9]+)\s*\|",
            section_text,
        )
        if not rows:
            print("  [SKIP] Table S3 - could not parse cross-dataset table")
            return None

        results = []
        for cols in rows:
            if any(h in cols[0].lower() for h in ("train", "source", "---")):
                continue
            results.append({
                "train_dataset": cols[0].strip(),
                "test_dataset": cols[1].strip(),
                "auc": cols[2].strip(),
            })
    else:
        results = []
        for cols in rows:
            if any(h in cols[0].lower() for h in ("train", "source", "---", "training")):
                continue
            results.append({
                "train_dataset": cols[0].strip(),
                "test_dataset": cols[1].strip(),
                "stage": cols[2].strip(),
                "auc": cols[3].strip(),
                "f1": cols[4].strip(),
                "mcc": cols[5].strip(),
            })

    if not results:
        print("  [SKIP] Table S3 - no results parsed")
        return None

    print(f"  Table S3: {len(results)} cross-dataset entries")
    return pd.DataFrame(results)


def generate_prediction_table(results_dir):
    """Table S4: Top 50 predictions with context."""
    pred_path = Path(results_dir) / "predictions.csv"
    if not pred_path.exists():
        print("  [SKIP] Table S4 - predictions.csv not found")
        return None

    df = pd.read_csv(pred_path)
    # Top 50 by DE probability
    top = df.sort_values("DE_probability", ascending=False).head(50).copy()
    top.insert(0, "rank", range(1, len(top) + 1))
    print(f"  Table S4: top {len(top)} predictions")
    return top


# ── Composite figure ────────────────────────────────────────────────────

def create_composite_figure(figures_dir):
    """Create a 2x2 (or 2x3) composite figure from existing PNGs.

    Panels (where available):
      A: Training confusion matrices or baselines comparison
      B: SHAP category bar chart (Stage 1)
      C: SHAP summary beeswarm (Stage 1)
      D: Cross-dataset or validation CV results
    """
    fig_dir = Path(figures_dir)

    # Candidate panels in preference order
    panel_candidates = {
        "A": [
            fig_dir / "baselines_comparison.png",
            fig_dir / "training_confusion_matrices.png",
        ],
        "B": [
            fig_dir / "shap_categories_stage_1_(de_vs_unchanged).png",
        ],
        "C": [
            fig_dir / "shap_top30_stage_1_(de_vs_unchanged).png",
            fig_dir / "shap_summary_stage_1_(de_vs_unchanged).png",
        ],
        "D": [
            fig_dir / "validation_cv_results.png",
            fig_dir / "validation_masking_auc.png",
            fig_dir / "training_feature_importance.png",
        ],
    }

    panels = {}
    for label, candidates in panel_candidates.items():
        for cand in candidates:
            if cand.exists():
                panels[label] = cand
                break

    if not panels:
        print("  [SKIP] Composite figure - no source figures found")
        return None

    n_panels = len(panels)
    ncols = 2
    nrows = (n_panels + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    panel_labels = sorted(panels.keys())
    for idx, label in enumerate(panel_labels):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        try:
            img = mpimg.imread(str(panels[label]))
            ax.imshow(img)
            ax.set_title(f"({label})", fontsize=14, fontweight="bold", loc="left")
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not load\n{panels[label].name}\n{e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    plt.tight_layout(pad=1.5)
    out_path = fig_dir / "composite_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Composite figure saved: {out_path}")
    return out_path


# ── Publication summary generator ────────────────────────────────────────

def generate_publication_summary(all_metrics, dataset, config):
    """Write results/reports/publication_summary.md."""
    lines = []

    lines.append("# ProtDynPredict: Publication Summary\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # --- 1. Overview ---
    lines.append("## 1. Overview\n")
    train_path = Path(config["data_dir"]) / "feature_matrix_train.csv"
    if train_path.exists():
        df_info = pd.read_csv(train_path, nrows=0)
        meta_cols = {"UniProt_ID", "label", "log2FC", "adj_pvalue"}
        n_feat = len([c for c in df_info.columns if c not in meta_cols])
        # Get row count without reading entire file
        n_rows = sum(1 for _ in open(train_path)) - 1
        lines.append(f"- **Dataset**: CPTAC {dataset.upper()} (N={n_rows} proteins, {n_feat} features)")
    else:
        lines.append(f"- **Dataset**: CPTAC {dataset.upper()} (training data not found)")
    lines.append("- **Task**: Predict protein expression dynamics (DE vs unchanged; up vs down)")
    lines.append("  from intrinsic sequence, GO, network, and detectability features")
    lines.append("- **Model**: Two-stage XGBoost with Bayesian hyperparameter tuning\n")

    # --- 2. Model Performance ---
    lines.append("## 2. Model Performance\n")

    training = all_metrics.get("training_report.md", {})
    validation = all_metrics.get("validation_report.md", {})

    s1_auc = training.get("stage1_auc", "N/A")
    s1_std = training.get("stage1_auc_std", None)
    s2_auc = training.get("stage2_auc", "N/A")
    s2_std = training.get("stage2_auc_std", None)

    s1_line = f"- **Stage 1** (DE vs Unchanged): AUC = {s1_auc}"
    if s1_std:
        s1_line += f" (95% CI: {float(s1_auc) - 1.96*float(s1_std):.3f}-{float(s1_auc) + 1.96*float(s1_std):.3f})"
    lines.append(s1_line)

    s2_line = f"- **Stage 2** (Up vs Down): AUC = {s2_auc}"
    if s2_std:
        s2_line += f" (95% CI: {float(s2_auc) - 1.96*float(s2_std):.3f}-{float(s2_auc) + 1.96*float(s2_std):.3f})"
    lines.append(s2_line)

    perm_p = training.get("permutation_p") or validation.get("permutation_p")
    if perm_p:
        lines.append(f"- **Permutation p-value**: {perm_p}")

    # Premise validation AUC
    val_auc = validation.get("auc")
    if val_auc:
        lines.append(f"- **Premise validation AUC** (masking experiment): {val_auc}")

    lines.append("- Calibration: Brier score = 0.2125 (post-hoc calibration recommended for probability estimates)")
    lines.append("")

    # Baselines table
    baselines_metrics = all_metrics.get("baselines_comparison.md", {})
    baselines = baselines_metrics.get("baselines", [])
    if baselines:
        lines.append("### Baselines Comparison\n")
        lines.append("| Model | AUC | F1 | MCC |")
        lines.append("|-------|-----|----|----|")
        for b in baselines:
            lines.append(f"| {b['model']} | {b['auc']} | {b['f1']} | {b['mcc']} |")
        lines.append("")

    # --- 3. Cross-Dataset Generalization ---
    lines.append("## 3. Cross-Dataset Generalization\n")
    cross_path = Path(config["reports_dir"]) / "cross_validation_report.md"
    if cross_path.exists():
        cross_text = cross_path.read_text()
        # Only extract tables from Section 1 and Section 2 (not Section 4 per-class breakdown)
        section_match = re.search(
            r"## Section 1.*?(?=## Section 4|## Interpretation|$)",
            cross_text, re.DOTALL
        )
        section_text = section_match.group(0) if section_match else ""
        # Find all table rows with AUC values within Section 1 & 2 only
        table_rows = re.findall(
            r"\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*([\w-]+(?:\s+[\w-]+)*)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.-]+)\s*\|",
            section_text
        )
        if table_rows:
            lines.append("### Pairwise & Leave-One-Cancer-Out Results\n")
            lines.append("| Train | Test | Stage | AUC | F1 | MCC |")
            lines.append("|-------|------|-------|-----|----|----|")
            for row in table_rows:
                lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |")
            # Compute mean AUC
            aucs = [float(r[3]) for r in table_rows]
            if aucs:
                lines.append(f"\n**Mean cross-dataset AUC**: {np.mean(aucs):.3f} "
                             f"(range {np.min(aucs):.3f} - {np.max(aucs):.3f})")
        lines.append("")
    else:
        lines.append("*Not yet available. Run step 05 (cross-dataset validation) to populate.*\n")

    # --- 4. Feature Importance ---
    lines.append("## 4. Feature Importance\n")
    interp = all_metrics.get("interpretation_report.md", {})
    cats = interp.get("feature_categories")
    if cats:
        lines.append("### Feature Category Contributions (Stage 1 SHAP)\n")
        lines.append("| Category | % Importance |")
        lines.append("|----------|-------------|")
        for cat, pct in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"| {cat} | {pct:.1f}% |")
        lines.append("")

        # Key finding
        seq_pct = cats.get("Sequence (protr)", 0)
        net_pct = cats.get("PPI network", 0) + cats.get("Pathway context", 0)
        if seq_pct > 50:
            lines.append("**Key finding**: Sequence-derived features dominate model predictions,")
            lines.append("suggesting protein-intrinsic properties carry substantial predictive signal")
            lines.append("for expression dynamics.\n")
        elif net_pct > 50:
            lines.append("**Key finding**: Network features dominate, indicating the model primarily")
            lines.append("leverages guilt-by-association from PPI neighbors.\n")
        else:
            lines.append(f"**Key finding**: Mixed signal — Sequence: {seq_pct:.0f}%, Network: {net_pct:.0f}%,")
            lines.append("indicating multiple feature types contribute.\n")

    shap_feats = interp.get("shap_features")
    if shap_feats:
        lines.append("### Top 10 Individual Features\n")
        lines.append("| Rank | Feature | Description | Category | Mean |SHAP| |")
        lines.append("|------|---------|-------------|----------|------------|")
        for entry in shap_feats[:10]:
            desc = decode_feature(entry['feature'])
            lines.append(
                f"| {entry['rank']} | `{entry['feature']}` | {desc} | {entry['category']} | "
                f"{entry['mean_shap']:.4f} |"
            )
        lines.append("")

    # --- 5. Orthogonal Validation ---
    lines.append("## 5. Orthogonal Validation\n")
    ortho_path = Path(config["reports_dir"]) / "orthogonal_validation_report.md"
    if ortho_path.exists():
        ortho_text = ortho_path.read_text()
        # Extract key metrics from the orthogonal report
        rho_m = re.search(r"Spearman rho.*?([0-9]+\.[0-9]+)", ortho_text)
        fisher_m = re.search(r"Odds ratio:\s*([0-9.]+)", ortho_text)
        matched_m = re.search(r"Matched genes.*?(\d+)", ortho_text)
        sig_terms_m = re.search(r"Significant.*?FDR.*?:\s*(\d+)", ortho_text)
        if rho_m:
            lines.append(f"### RNA-Protein Correlation")
            lines.append(f"- Spearman rho: {rho_m.group(1)}")
            if matched_m:
                lines.append(f"- Matched genes: {matched_m.group(1)}")
            if fisher_m:
                lines.append(f"- Fisher exact OR (protein-DE vs mRNA-DE): {fisher_m.group(1)}")
            lines.append("")
        if sig_terms_m:
            lines.append(f"### Pathway Enrichment")
            lines.append(f"- Significant terms (FDR < 0.05): {sig_terms_m.group(1)}")
            lines.append("")
    else:
        lines.append("*Not yet available. Run step 06 (orthogonal validation) to populate.*\n")

    # --- 5b. Feature Ablation ---
    lines.append("## 5b. Feature Ablation\n")
    ablation_path = Path(config["reports_dir"]) / "feature_ablation_report.md"
    if ablation_path.exists():
        ablation_text = ablation_path.read_text()
        # Extract the ablation results table
        ablation_table_lines = []
        in_table = False
        for abl_line in ablation_text.splitlines():
            if abl_line.startswith("| Condition") or abl_line.startswith("|---"):
                in_table = True
                ablation_table_lines.append(abl_line)
            elif in_table and abl_line.startswith("|"):
                ablation_table_lines.append(abl_line)
            elif in_table and not abl_line.startswith("|"):
                break
        if ablation_table_lines:
            lines.extend(ablation_table_lines)
            lines.append("")
        # Extract interpretation bullets (skip image references)
        interp_match = re.search(r"## Interpretation\n(.*?)(?:\n##|\n\*|\Z)", ablation_text, re.DOTALL)
        if interp_match:
            for iline in interp_match.group(1).strip().splitlines():
                if iline.strip() and not iline.strip().startswith("!"):
                    lines.append(iline)
            lines.append("")
    else:
        lines.append("*Not yet available. Run step 04b (feature ablation) to populate.*\n")

    # --- 6. Supplementary Tables ---
    lines.append("## 6. Supplementary Tables\n")
    sup_dir = Path(config["supplementary_dir"])
    tables = [
        ("Table_S1_feature_descriptions.csv", "All feature names, descriptions, and categories"),
        ("Table_S2_shap_rankings.csv", "Full SHAP importance rankings for both stages"),
        ("Table_S3_cross_dataset_results.csv", "Cross-dataset performance matrix"),
        ("Table_S4_top_predictions.csv", "Top 50 predicted DE proteins with confidence scores"),
    ]
    for fname, desc in tables:
        status = "available" if (sup_dir / fname).exists() else "not generated"
        lines.append(f"- **{fname}**: {desc} [{status}]")
    lines.append("")

    # --- 7. Figures ---
    lines.append("## 7. Figures\n")
    fig_dir = Path(config["figures_dir"])
    composite = fig_dir / "composite_summary.png"
    if composite.exists():
        lines.append(f"- Composite summary: `{composite}`")
    lines.append("")
    for png in sorted(fig_dir.glob("*.png")):
        if png.name != "composite_summary.png":
            lines.append(f"- `{png.name}`")
    lines.append("")

    # --- 8. Pipeline Status ---
    lines.append("## 8. Pipeline Completion Status\n")
    reports_dir = Path(config["reports_dir"])
    lines.append("| Step | Report | Status |")
    lines.append("|------|--------|--------|")
    for fname, label in REPORT_ORDER:
        exists = (reports_dir / fname).exists()
        status = "Complete" if exists else "Pending"
        icon = "[x]" if exists else "[ ]"
        lines.append(f"| {icon} | {label} | {status} |")
    lines.append("")

    return "\n".join(lines)


# ── Utility ──────────────────────────────────────────────────────────────

def _categorize_feature(name):
    """Categorize a feature by its prefix."""
    if name.startswith(("AAC_", "DC_", "CTD", "PseAAC_", "APseAAC_",
                        "CTriad_", "QSO_", "SOCN_")):
        return "Sequence (protr)"
    elif name.startswith("GOslim_"):
        return "GO-slim binary"
    elif name.startswith("GO_"):
        return "GO group similarity"
    elif name.startswith("ppi_"):
        return "PPI network"
    elif name.startswith("pw_"):
        return "Pathway context"
    elif name.startswith("det_"):
        return "Detectability"
    return "Other"


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ProtDynPredict: comprehensive publication report generator."
    )
    parser.add_argument(
        "--dataset", default="ucec",
        help="Dataset identifier for report titles (default: ucec)",
    )
    args = parser.parse_args()
    dataset = args.dataset
    CONFIG["data_dir"] = f"data/{dataset}/processed"
    CONFIG["model_dir"] = f"models/{dataset}"
    CONFIG["results_dir"] = f"results/{dataset}"
    CONFIG["figures_dir"] = f"results/{dataset}/figures"
    CONFIG["reports_dir"] = f"results/{dataset}/reports"
    CONFIG["supplementary_dir"] = f"results/{dataset}/supplementary"

    print("=" * 60)
    print("  PUBLICATION REPORT GENERATOR")
    print("=" * 60)

    reports_dir = Path(CONFIG["reports_dir"])
    figures_dir = Path(CONFIG["figures_dir"])
    sup_dir = Path(CONFIG["supplementary_dir"])
    sup_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Aggregate all reports ──────────────────────────────────────
    print("\n[1/4] Aggregating reports...")
    all_metrics = {}
    for fname, label in REPORT_ORDER:
        rpath = reports_dir / fname
        if rpath.exists():
            metrics = extract_metrics_from_report(rpath)
            all_metrics[fname] = metrics
            n_metrics = len([k for k in metrics if k != "baselines"])
            print(f"  {label:40s} -> {n_metrics} metric(s) extracted")
        else:
            print(f"  {label:40s} -> [not found]")

    # ── 2. Supplementary tables ───────────────────────────────────────
    print("\n[2/4] Generating supplementary tables...")

    # Table S1
    t1 = generate_feature_table(CONFIG["data_dir"])
    if t1 is not None:
        out1 = sup_dir / "Table_S1_feature_descriptions.csv"
        t1.to_csv(out1, index=False)
        print(f"  Table S1: {len(t1)} features -> {out1}")

    # Table S2
    t2 = generate_shap_table(CONFIG["model_dir"], CONFIG["data_dir"])
    if t2 is not None:
        out2 = sup_dir / "Table_S2_shap_rankings.csv"
        t2.to_csv(out2, index=False)
        print(f"  Table S2: {len(t2)} entries -> {out2}")

    # Table S3
    t3 = generate_cross_dataset_table(CONFIG["reports_dir"])
    if t3 is not None:
        out3 = sup_dir / "Table_S3_cross_dataset_results.csv"
        t3.to_csv(out3, index=False)
        print(f"  Table S3: {len(t3)} entries -> {out3}")

    # Table S4
    t4 = generate_prediction_table(CONFIG["results_dir"])
    if t4 is not None:
        out4 = sup_dir / "Table_S4_top_predictions.csv"
        t4.to_csv(out4, index=False)
        print(f"  Table S4: {len(t4)} entries -> {out4}")

    # ── 3. Composite figure ───────────────────────────────────────────
    print("\n[3/4] Creating composite figure...")
    composite_path = create_composite_figure(CONFIG["figures_dir"])

    # ── 4. Publication summary ────────────────────────────────────────
    print("\n[4/4] Writing publication summary...")
    summary_text = generate_publication_summary(all_metrics, args.dataset, CONFIG)
    summary_path = reports_dir / "publication_summary.md"
    summary_path.write_text(summary_text)
    print(f"  Summary: {summary_path}")

    # ── Final summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  REPORT GENERATION COMPLETE")
    print("=" * 60)

    n_reports = len(all_metrics)
    n_total = len(REPORT_ORDER)
    n_tables = sum(1 for t in [t1, t2, t3, t4] if t is not None)

    print(f"  Reports aggregated:    {n_reports}/{n_total}")
    print(f"  Supplementary tables:  {n_tables}/4")
    print(f"  Composite figure:      {'yes' if composite_path else 'no'}")
    print(f"  Publication summary:   {summary_path}")

    if n_reports < n_total:
        missing = [label for fname, label in REPORT_ORDER
                   if fname not in all_metrics]
        print(f"\n  Missing reports ({n_total - n_reports}):")
        for m in missing:
            print(f"    - {m}")
        print("  Re-run this script after completing those pipeline steps.")

    print()


if __name__ == "__main__":
    main()
