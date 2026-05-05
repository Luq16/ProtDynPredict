#!/usr/bin/env python3
"""
04b_feature_ablation.py
Feature ablation analysis for ProtDynPredict.

Removes one feature category at a time and measures AUC drop relative to
the full-feature baseline under 5-fold stratified CV.

Input:  data/<dataset>/processed/feature_matrix_train.csv
Output: results/<dataset>/figures/feature_ablation*.png
        results/<dataset>/reports/feature_ablation*_report.md
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
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
    """5-fold stratified CV mean AUC. Returns (mean, std, fold_aucs)."""
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
    fold_aucs = np.array(aucs)
    return float(np.mean(fold_aucs)), float(np.std(fold_aucs)), fold_aucs


def get_stage_data(df, all_features, stage):
    """Return stage-specific matrix, labels, and display metadata."""
    X_all = df[all_features].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    if stage == 1:
        y = np.where(df["label"] == "unchanged", 0, 1)
        counts = f"DE: {np.sum(y==1)} | Unchanged: {np.sum(y==0)}"
        stage_slug = ""
        stage_label = "Stage 1: DE vs Unchanged"
    else:
        de_mask = df["label"].isin(["up", "down"]).values
        X_all = X_all[de_mask]
        y = np.where(df.loc[de_mask, "label"] == "up", 1, 0)
        counts = f"Up: {np.sum(y==1)} | Down: {np.sum(y==0)}"
        stage_slug = "_stage2"
        stage_label = "Stage 2: Up vs Down"

    return X_all, y, counts, stage_slug, stage_label


def main():
    parser = argparse.ArgumentParser(description="Feature ablation analysis")
    parser.add_argument("--dataset", default="ucec")
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    ds = args.dataset
    stage = args.stage

    data_path = Path(f"data/{ds}/processed/feature_matrix_train.csv")
    fig_dir = Path(f"results/{ds}/figures")
    rpt_dir = Path(f"results/{ds}/reports")
    fig_dir.mkdir(parents=True, exist_ok=True)
    rpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  Feature Ablation Analysis [{ds.upper()}] - Stage {stage}")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    all_features = _safe_features(df.columns)
    X_all, y, counts, stage_slug, stage_label = get_stage_data(df, all_features, stage)

    print(f"  Proteins: {len(y)} | Features: {len(all_features)}")
    print(f"  {counts}\n")

    # ── Baseline (all features) ──────────────────────────────────────────
    print("Training baseline (all features)...")
    base_auc, base_std, base_folds = cv_auc(X_all, y)
    print(f"  Baseline AUC: {base_auc:.4f} +/- {base_std:.4f}\n")

    # ── Ablation per category ────────────────────────────────────────────
    # results: (name, auc, std, delta, n_features, p_value)
    results = [("All features", base_auc, base_std, 0.0, len(all_features), None)]

    for cat_name, prefixes in CATEGORIES.items():
        keep_idx = [i for i, c in enumerate(all_features)
                    if not _matches_category(c, prefixes)]
        n_removed = len(all_features) - len(keep_idx)
        if n_removed == 0:
            print(f"  {cat_name}: no matching features, skipping")
            continue

        X_abl = X_all[:, keep_idx]
        print(f"Training without {cat_name} ({n_removed} features removed)...")
        auc, std, fold_aucs = cv_auc(X_abl, y)
        delta = auc - base_auc

        # Paired t-test: compare per-fold AUCs to baseline
        t_stat, p_val = ttest_rel(base_folds, fold_aucs)
        print(f"  AUC: {auc:.4f} +/- {std:.4f}  (delta {delta:+.4f}, "
              f"paired t-test p={p_val:.4f})")
        results.append((f"w/o {cat_name}", auc, std, delta, len(keep_idx), p_val))

    # ── Bar chart ────────────────────────────────────────────────────────
    labels = [r[0] for r in results]
    aucs = [r[1] for r in results]
    stds = [r[2] for r in results]

    colors = ["#2c7bb6"] + ["#d7191c" if (r[3] < -0.005 and r[5] is not None and r[5] < 0.05)
                             else "#fdae61" for r in results[1:]]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    ax.bar(x, aucs, yerr=stds, capsize=4, color=colors,
           edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean AUC (5-fold CV)")
    ax.set_title(f"Feature Ablation - {stage_label} [{ds.upper()}]")
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
    fig_path = fig_dir / f"feature_ablation{stage_slug}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Figure saved: {fig_path}")

    # ── Markdown report ──────────────────────────────────────────────────
    lines = [
        "# Feature Ablation Report\n",
        f"**Dataset**: {ds.upper()}",
        f"**Stage**: {stage_label}",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Method\n",
        "Each feature category is removed in turn from the safe feature set.",
        f"A 5-fold stratified CV XGBoost ({stage_label}) is trained",
        "and the mean AUC compared to the full-feature baseline.\n",
        "## Results\n",
        "| Condition | # Features | AUC (mean +/- std) | Delta AUC | Paired t-test p |",
        "|-----------|------------|--------------------|-----------:|----------------:|",
    ]
    for name, auc, std, delta, nf, pval in results:
        pval_str = "—" if pval is None else f"{pval:.4f}"
        sig_str = ""
        if pval is not None and pval < 0.05:
            sig_str = " *"
        lines.append(f"| {name} | {nf} | {auc:.4f} +/- {std:.4f} | "
                     f"{delta:+.4f} | {pval_str}{sig_str} |")

    lines.append("\n\\* p < 0.05 (statistically significant difference from baseline)\n")
    lines.append("## Interpretation\n")
    # Sort ablations by impact (most negative delta first)
    ablations = [r for r in results if r[0] != "All features"]
    ablations.sort(key=lambda r: r[3])
    if ablations:
        worst = ablations[0]
        lines.append(f"- **Most impactful category**: {worst[0]} "
                     f"(AUC drop {worst[3]:+.4f}, p={worst[5]:.4f})")
        best = ablations[-1]
        lines.append(f"- **Least impactful category**: {best[0]} "
                     f"(AUC change {best[3]:+.4f}, p={best[5]:.4f})")
        sig_ablations = [r for r in ablations if r[5] is not None and r[5] < 0.05]
        if sig_ablations:
            lines.append(f"- **Statistically significant ablations** (paired t-test, "
                         f"p < 0.05): {', '.join(r[0] for r in sig_ablations)}")
        else:
            lines.append("- No ablation showed a statistically significant AUC "
                         "difference from baseline (paired t-test, p < 0.05)")

    fig_rel = f"../figures/feature_ablation{stage_slug}.png"
    lines.append(f"\n![Feature Ablation]({fig_rel})\n")
    lines.append("*Generated by `04b_feature_ablation.py`*\n")

    rpt_path = rpt_dir / f"feature_ablation{stage_slug}_report.md"
    with open(rpt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {rpt_path}")

    print("\n" + "=" * 60)
    print("  FEATURE ABLATION COMPLETE")
    print(f"  {stage_label}")
    print(f"  Baseline AUC: {base_auc:.4f}")
    for name, auc, std, delta, nf, pval in results[1:]:
        pval_str = f", p={pval:.4f}" if pval is not None else ""
        print(f"  {name}: {auc:.4f} ({delta:+.4f}{pval_str})")
    print("=" * 60)


if __name__ == "__main__":
    main()
