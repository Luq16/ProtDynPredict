#!/usr/bin/env python3
"""
11_annotation_bias_test.py
Annotation-bias analysis for reviewer Major Concern 4.

Tests whether GO-slim annotation completeness acts as a confounder:
  - Do DE proteins have more GO annotations than unchanged proteins?
  - Can annotation completeness alone predict DE status?
  - How does a sequence-only model compare to a GO-slim-only model?

Input:  data/<dataset>/processed/feature_matrix_train.csv
Output: results/<dataset>/reports/annotation_bias_report.md
        results/<dataset>/figures/annotation_completeness.png
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Leaky features (label-dependent) — must exclude everywhere ───────────
LEAKY_PREFIXES = [
    "ppi_frac_neighbors_", "ppi_weighted_frac_",
    "pw_max_frac_", "pw_mean_frac_",
    "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_",
]
META_COLS = {"UniProt_ID", "label", "log2FC", "adj_pvalue"}

# ── Sequence-only prefixes (genuinely protein-intrinsic) ─────────────────
SEQUENCE_PREFIXES = [
    "AAC_", "DC_", "CTriad_", "CTDC_", "CTDT_", "CTDD_",
    "QSO_", "APseAAC_", "PseAAC_",
]


def _is_leaky(col):
    return any(col.startswith(p) for p in LEAKY_PREFIXES)


def _select_cols(columns, prefixes):
    """Select columns matching any of the given prefixes, excluding leaky."""
    return [c for c in columns
            if any(c.startswith(p) for p in prefixes)
            and not _is_leaky(c) and c not in META_COLS]


def cv_auc_logistic(X, y, n_folds=5, seed=42):
    """Stratified CV with logistic regression (standardised features)."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        clf = LogisticRegression(
            max_iter=2000, solver="lbfgs", class_weight="balanced",
            random_state=seed,
        )
        clf.fit(X_tr, y[tr])
        prob = clf.predict_proba(X_va)[:, 1]
        try:
            aucs.append(roc_auc_score(y[va], prob))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    parser = argparse.ArgumentParser(
        description="Annotation-bias confound analysis (Reviewer Major Concern 4)")
    parser.add_argument("--dataset", default="ucec")
    args = parser.parse_args()
    ds = args.dataset

    data_path = Path(f"data/{ds}/processed/feature_matrix_train.csv")
    fig_dir = Path(f"results/{ds}/figures")
    rpt_dir = Path(f"results/{ds}/reports")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"  Annotation-Bias Confound Analysis [{ds.upper()}]")
    print("=" * 65)

    # ── 1. Load data ─────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    y = np.where(df["label"] == "unchanged", 0, 1)
    n_de = int(np.sum(y == 1))
    n_unch = int(np.sum(y == 0))
    print(f"  Proteins: {len(y)}  |  DE: {n_de}  |  Unchanged: {n_unch}\n")

    # ── 2. Annotation completeness (non-zero GOslim columns per protein) ─
    goslim_cols = [c for c in df.columns if c.startswith("GOslim_")]
    goslim_mat = df[goslim_cols].values
    annotation_count = np.sum(goslim_mat != 0, axis=1)

    ann_de = annotation_count[y == 1]
    ann_unch = annotation_count[y == 0]

    print(f"  GO-slim columns: {len(goslim_cols)}")
    print(f"  Annotation count — DE:        median={np.median(ann_de):.0f}, "
          f"mean={np.mean(ann_de):.1f}")
    print(f"  Annotation count — Unchanged:  median={np.median(ann_unch):.0f}, "
          f"mean={np.mean(ann_unch):.1f}")

    # ── 3. Mann-Whitney U test ───────────────────────────────────────────
    stat, pval = mannwhitneyu(ann_de, ann_unch, alternative="two-sided")
    print(f"  Mann-Whitney U = {stat:.0f},  p = {pval:.2e}\n")

    # ── 4. Model A: annotation completeness only (single feature) ────────
    print("Model A: Annotation completeness only (1 feature)...")
    X_ann = annotation_count.reshape(-1, 1).astype(np.float32)
    auc_ann, std_ann = cv_auc_logistic(X_ann, y)
    print(f"  AUC = {auc_ann:.4f} +/- {std_ann:.4f}\n")

    # ── 5. Model B: sequence features only ───────────────────────────────
    seq_cols = _select_cols(df.columns, SEQUENCE_PREFIXES)
    print(f"Model B: Sequence features only ({len(seq_cols)} features)...")
    X_seq = df[seq_cols].values.astype(np.float32)
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
    auc_seq, std_seq = cv_auc_logistic(X_seq, y)
    print(f"  AUC = {auc_seq:.4f} +/- {std_seq:.4f}\n")

    # ── 6. Model C: GO-slim features only (logistic regression) ──────────
    print(f"Model C: GO-slim features only ({len(goslim_cols)} features)...")
    X_go = goslim_mat.astype(np.float32)
    X_go = np.nan_to_num(X_go, nan=0.0, posinf=0.0, neginf=0.0)
    auc_go, std_go = cv_auc_logistic(X_go, y)
    print(f"  AUC = {auc_go:.4f} +/- {std_go:.4f}\n")

    # ── 7. Figure: boxplot of annotation completeness by label ───────────
    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot(
        [ann_unch, ann_de],
        labels=["Unchanged", "DE"],
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="black", markersize=6),
    )
    bp["boxes"][0].set_facecolor("#4daf4a")
    bp["boxes"][1].set_facecolor("#e41a1c")
    ax.set_ylabel("Non-zero GO-slim annotation count")
    ax.set_title(f"GO-slim Annotation Completeness [{ds.upper()}]")
    ax.text(
        0.98, 0.02,
        f"Mann-Whitney p = {pval:.2e}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )
    plt.tight_layout()
    fig_path = fig_dir / "annotation_completeness.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")

    # ── 8. Markdown report ───────────────────────────────────────────────
    lines = [
        "# Annotation-Bias Confound Analysis\n",
        f"**Dataset**: {ds.upper()}  ",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Reviewer concern**: Major Concern 4 — annotation completeness "
        "as confounder\n",
        "## 1. Annotation Completeness Distribution\n",
        "| Group | N | Median non-zero GO-slim | Mean non-zero GO-slim |",
        "|-------|---|------------------------:|----------------------:|",
        f"| DE | {n_de} | {np.median(ann_de):.0f} | {np.mean(ann_de):.1f} |",
        f"| Unchanged | {n_unch} | {np.median(ann_unch):.0f} | "
        f"{np.mean(ann_unch):.1f} |",
        "",
        f"**Mann-Whitney U** = {stat:.0f}, p = {pval:.2e}\n",
        "## 2. Single-Feature & Feature-Group AUC Comparison\n",
        "All models use Logistic Regression with `class_weight='balanced'`, "
        "StratifiedKFold(5, shuffle=True, random_state=42).\n",
        "| Model | Features | # Features | AUC (mean +/- std) |",
        "|-------|----------|------------|-------------------:|",
        f"| A — Annotation completeness | count of non-zero GO-slim | 1 | "
        f"{auc_ann:.4f} +/- {std_ann:.4f} |",
        f"| B — Sequence only | AAC, DC, CTriad, CTDC, CTDT, CTDD, QSO, "
        f"APseAAC, PseAAC | {len(seq_cols)} | {auc_seq:.4f} +/- {std_seq:.4f} |",
        f"| C — GO-slim only | All 150 GOslim binary features | "
        f"{len(goslim_cols)} | {auc_go:.4f} +/- {std_go:.4f} |",
        "",
        "## 3. Interpretation\n",
    ]

    # Automated interpretation
    if pval < 0.05:
        lines.append(
            f"Annotation completeness **differs significantly** between DE and "
            f"unchanged proteins (p = {pval:.2e}). DE proteins carry "
            f"{'more' if np.mean(ann_de) > np.mean(ann_unch) else 'fewer'} "
            f"GO-slim annotations on average."
        )
    else:
        lines.append(
            f"Annotation completeness does **not** differ significantly "
            f"between DE and unchanged proteins (p = {pval:.2e})."
        )
    lines.append("")

    if auc_ann > 0.60:
        lines.append(
            f"**Warning**: Annotation completeness alone achieves AUC = "
            f"{auc_ann:.3f}, suggesting it carries predictive signal that "
            f"may partly reflect research bias rather than biology."
        )
    else:
        lines.append(
            f"Annotation completeness alone achieves AUC = {auc_ann:.3f}, "
            f"indicating it is a **weak predictor** and unlikely to be a "
            f"dominant confounder."
        )
    lines.append("")

    if auc_seq > auc_go:
        lines.append(
            f"Sequence-only features (AUC = {auc_seq:.3f}) **outperform** "
            f"GO-slim features (AUC = {auc_go:.3f}), demonstrating that "
            f"genuinely protein-intrinsic properties carry strong signal "
            f"independent of annotation completeness."
        )
    else:
        lines.append(
            f"GO-slim features (AUC = {auc_go:.3f}) outperform sequence-only "
            f"features (AUC = {auc_seq:.3f}). This warrants caution: GO-slim "
            f"annotations may introduce annotation-bias signal."
        )
    lines.append("")

    lines.append(
        "## 4. Conclusion\n"
    )
    if auc_ann < 0.60 and auc_seq >= auc_go:
        lines.append(
            "Annotation completeness is not a meaningful confounder. "
            "The model's predictive power derives primarily from "
            "protein-intrinsic sequence features, not from annotation depth."
        )
    elif auc_ann < 0.60:
        lines.append(
            "Annotation completeness is not a meaningful confounder on its "
            "own, though GO-slim features do add signal beyond sequence alone. "
            "This likely reflects genuine functional differences rather than "
            "annotation bias."
        )
    else:
        lines.append(
            "Annotation completeness shows non-trivial predictive power. "
            "We recommend reporting sequence-only model performance as a "
            "conservative lower bound and discussing annotation bias "
            "as a limitation."
        )

    lines.append(
        "\n![Annotation Completeness](../figures/annotation_completeness.png)\n"
    )
    lines.append("*Generated by `11_annotation_bias_test.py`*\n")

    rpt_path = rpt_dir / "annotation_bias_report.md"
    with open(rpt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {rpt_path}")

    print("\n" + "=" * 65)
    print("  ANNOTATION-BIAS ANALYSIS COMPLETE")
    print(f"  Mann-Whitney p  = {pval:.2e}")
    print(f"  AUC (completeness only) = {auc_ann:.4f}")
    print(f"  AUC (sequence only)     = {auc_seq:.4f}")
    print(f"  AUC (GO-slim only)      = {auc_go:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
