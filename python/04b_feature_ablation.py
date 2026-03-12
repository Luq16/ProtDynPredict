#!/usr/bin/env python3
"""
04b_feature_ablation.py
Feature ablation analysis for ProtDynPredict.

Removes one feature category at a time and measures AUC drop relative to
the full-feature baseline (Stage 1: DE vs unchanged, 5-fold stratified CV).

Input:  data/<dataset>/processed/feature_matrix_train.csv
Output: results/<dataset>/figures/feature_ablation.png
        results/<dataset>/reports/feature_ablation_report.md
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Leaky features (label-dependent) ────────────────────────────────────
LEAKY_PREFIXES = [
    "ppi_frac_neighbors_", "ppi_weighted_frac_",
    "pw_max_frac_", "pw_mean_frac_",
    "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_",
]

# ── Feature categories by prefix ────────────────────────────────────────
CATEGORIES = {
    "Sequence": [
        "AAC_", "DC_", "CTriad_", "CTDC_", "CTDT_", "CTDD_",
        "QSO_", "APseAAC_", "PseAAC_",
    ],
    "GO-slim":       ["GOslim_"],
    "PPI network":   ["ppi_"],
    "Detectability": ["det_"],
    "Pathway":       ["pw_"],
}

# ── Default XGBoost parameters (no Optuna) ──────────────────────────────
XGB_PARAMS = dict(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


def _safe_features(columns):
    """Return feature columns excluding metadata and leaky features."""
    meta = {"UniProt_ID", "label", "log2FC", "adj_pvalue"}
    return [c for c in columns
            if c not in meta and not any(c.startswith(p) for p in LEAKY_PREFIXES)]


def _matches_category(col, prefixes):
    return any(col.startswith(p) for p in prefixes)


def cv_auc(X, y, n_folds=5, seed=42):
    """5-fold stratified CV mean AUC."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scale_pos = np.sum(y == 0) / max(np.sum(y == 1), 1)
    params = {**XGB_PARAMS, "scale_pos_weight": scale_pos}
    aucs = []
    for tr, va in cv.split(X, y):
        clf = xgb.XGBClassifier(**params)
        clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        prob = clf.predict_proba(X[va])[:, 1]
        try:
            aucs.append(roc_auc_score(y[va], prob))
        except ValueError:
            aucs.append(0.5)
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    parser = argparse.ArgumentParser(description="Feature ablation analysis")
    parser.add_argument("--dataset", default="ucec")
    args = parser.parse_args()
    ds = args.dataset

    data_path = Path(f"data/{ds}/processed/feature_matrix_train.csv")
    fig_dir = Path(f"results/{ds}/figures")
    rpt_dir = Path(f"results/{ds}/reports")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Feature Ablation Analysis [{ds.upper()}]")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    all_features = _safe_features(df.columns)
    X_all = df[all_features].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.where(df["label"] == "unchanged", 0, 1)

    print(f"  Proteins: {len(y)} | Features: {len(all_features)}")
    print(f"  DE: {np.sum(y==1)} | Unchanged: {np.sum(y==0)}\n")

    # ── Baseline (all features) ──────────────────────────────────────────
    print("Training baseline (all features)...")
    base_auc, base_std = cv_auc(X_all, y)
    print(f"  Baseline AUC: {base_auc:.4f} +/- {base_std:.4f}\n")

    # ── Ablation per category ────────────────────────────────────────────
    results = [("All features", base_auc, base_std, 0.0, len(all_features))]

    for cat_name, prefixes in CATEGORIES.items():
        keep_idx = [i for i, c in enumerate(all_features)
                    if not _matches_category(c, prefixes)]
        n_removed = len(all_features) - len(keep_idx)
        if n_removed == 0:
            print(f"  {cat_name}: no matching features, skipping")
            continue

        X_abl = X_all[:, keep_idx]
        print(f"Training without {cat_name} ({n_removed} features removed)...")
        auc, std = cv_auc(X_abl, y)
        delta = auc - base_auc
        print(f"  AUC: {auc:.4f} +/- {std:.4f}  (delta {delta:+.4f})")
        results.append((f"w/o {cat_name}", auc, std, delta, len(keep_idx)))

    # ── Bar chart ────────────────────────────────────────────────────────
    labels = [r[0] for r in results]
    aucs = [r[1] for r in results]
    stds = [r[2] for r in results]

    colors = ["#2c7bb6"] + ["#d7191c" if r[3] < -0.005 else "#fdae61"
                             for r in results[1:]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, aucs, yerr=stds, capsize=4, color=colors,
                  edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean AUC (5-fold CV)")
    ax.set_title(f"Feature Ablation — Stage 1 DE vs Unchanged [{ds.upper()}]")
    ax.axhline(base_auc, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    # annotate deltas
    for i, r in enumerate(results):
        if i == 0:
            ax.text(i, r[1] + r[2] + 0.005, f"{r[1]:.3f}", ha="center",
                    va="bottom", fontsize=8, fontweight="bold")
        else:
            ax.text(i, r[1] + r[2] + 0.005, f"{r[3]:+.3f}", ha="center",
                    va="bottom", fontsize=8)

    ymin = min(aucs) - max(stds) - 0.04
    ax.set_ylim(max(0.4, ymin), min(1.0, max(aucs) + max(stds) + 0.04))
    plt.tight_layout()
    fig_path = fig_dir / "feature_ablation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    # ── Markdown report ──────────────────────────────────────────────────
    lines = [
        "# Feature Ablation Report\n",
        f"**Dataset**: {ds.upper()}",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Method\n",
        "Each feature category is removed in turn from the safe feature set.",
        "A 5-fold stratified CV XGBoost (Stage 1: DE vs unchanged) is trained",
        "and the mean AUC compared to the full-feature baseline.\n",
        "## Results\n",
        "| Condition | # Features | AUC (mean +/- std) | Delta AUC |",
        "|-----------|------------|--------------------|-----------:|",
    ]
    for name, auc, std, delta, nf in results:
        lines.append(f"| {name} | {nf} | {auc:.4f} +/- {std:.4f} | "
                     f"{delta:+.4f} |")

    lines.append("\n## Interpretation\n")
    # Sort ablations by impact (most negative delta first)
    ablations = [r for r in results if r[0] != "All features"]
    ablations.sort(key=lambda r: r[3])
    if ablations:
        worst = ablations[0]
        lines.append(f"- **Most impactful category**: {worst[0]} "
                     f"(AUC drop {worst[3]:+.4f})")
        best = ablations[-1]
        lines.append(f"- **Least impactful category**: {best[0]} "
                     f"(AUC change {best[3]:+.4f})")

    lines.append("\n![Feature Ablation](../figures/feature_ablation.png)\n")
    lines.append("*Generated by `04b_feature_ablation.py`*\n")

    rpt_path = rpt_dir / "feature_ablation_report.md"
    with open(rpt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {rpt_path}")

    print("\n" + "=" * 60)
    print("  FEATURE ABLATION COMPLETE")
    print(f"  Baseline AUC: {base_auc:.4f}")
    for name, auc, std, delta, nf in results[1:]:
        print(f"  {name}: {auc:.4f} ({delta:+.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
