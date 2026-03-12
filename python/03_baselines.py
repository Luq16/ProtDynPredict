#!/usr/bin/env python3
"""
03_baselines.py
Compute all baseline models for comparison against XGBoost.

Baselines:
  1. Random prediction (respecting class priors)
  2. Majority class (everything = "unchanged")
  3. Nearest-neighbor by sequence similarity
  4. Label propagation (loads results from 02_label_propagation.py)

Input:  data/processed/feature_matrix_train.csv
Output: results/reports/baselines_comparison.md
        results/figures/baselines_comparison.png
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_baseline(name, clf_factory, X, y_encoded, classes, n_iter, mask_frac, rng):
    """Run a baseline with the same masking protocol."""
    results = []
    n_classes = len(classes)

    for i in range(n_iter):
        mask_idx, train_idx = [], []
        for cls in range(n_classes):
            idx = np.where(y_encoded == cls)[0]
            n_mask = max(1, int(len(idx) * mask_frac))
            rng.shuffle(idx)
            mask_idx.extend(idx[:n_mask])
            train_idx.extend(idx[n_mask:])

        mask_idx = np.array(mask_idx)
        train_idx = np.array(train_idx)

        clf = clf_factory()
        clf.fit(X[train_idx], y_encoded[train_idx])

        y_pred = clf.predict(X[mask_idx])
        y_true = y_encoded[mask_idx]

        try:
            y_proba = clf.predict_proba(X[mask_idx])
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except (ValueError, AttributeError):
            auc = 0.5

        f1 = f1_score(y_true, y_pred, average="macro")
        mcc = matthews_corrcoef(y_true, y_pred)
        results.append({"auc": auc, "f1": f1, "mcc": mcc})

    return {
        "name": name,
        "mean_auc": np.mean([r["auc"] for r in results]),
        "std_auc": np.std([r["auc"] for r in results]),
        "mean_f1": np.mean([r["f1"] for r in results]),
        "mean_mcc": np.mean([r["mcc"] for r in results]),
        "iterations": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Baselines comparison")
    parser.add_argument("--dataset", default="ucec", help="Dataset/cancer type (default: ucec)")
    args = parser.parse_args()
    dataset = args.dataset

    CONFIG = {
        "train_file": f"data/{dataset}/processed/feature_matrix_train.csv",
        "figures_dir": f"results/{dataset}/figures",
        "reports_dir": f"results/{dataset}/reports",
        "n_iterations": 5,
        "mask_fraction": 0.20,
        "random_state": 42,
    }

    print("=" * 60)
    print("  BASELINES COMPARISON")
    print(f"  Dataset: {dataset}")
    print("=" * 60)

    # Load data
    df = pd.read_csv(CONFIG["train_file"])
    meta_cols = ["UniProt_ID", "label", "log2FC", "adj_pvalue"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Remove leaky features (same filtering as 01_train_model.py)
    leaky_prefixes = [
        "ppi_frac_neighbors_", "ppi_weighted_frac_",
        "pw_max_frac_", "pw_mean_frac_",
        "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_"
    ]
    feature_cols = [c for c in feature_cols if not any(c.startswith(p) for p in leaky_prefixes)]

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    seed = CONFIG["random_state"]

    print(f"  Proteins: {X.shape[0]}, Classes: {dict(zip(classes, np.bincount(y_encoded)))}")

    # --- Run baselines (each with fresh RNG for reproducibility) ---
    all_baselines = []

    # 1. Random (stratified)
    print("\n1. Random prediction (stratified)...")
    result = run_baseline(
        "Random (stratified)",
        lambda: DummyClassifier(strategy="stratified", random_state=seed),
        X, y_encoded, classes, CONFIG["n_iterations"], CONFIG["mask_fraction"],
        np.random.RandomState(seed)
    )
    all_baselines.append(result)
    print(f"   AUC={result['mean_auc']:.3f}, F1={result['mean_f1']:.3f}")

    # 2. Majority class
    print("\n2. Majority class...")
    result = run_baseline(
        "Majority class",
        lambda: DummyClassifier(strategy="most_frequent"),
        X, y_encoded, classes, CONFIG["n_iterations"], CONFIG["mask_fraction"],
        np.random.RandomState(seed)
    )
    all_baselines.append(result)
    print(f"   AUC={result['mean_auc']:.3f}, F1={result['mean_f1']:.3f}")

    # 3. Nearest neighbor (k=1, sequence similarity proxy)
    print("\n3. Nearest neighbor (k=1)...")
    result = run_baseline(
        "1-NN (sequence similarity)",
        lambda: KNeighborsClassifier(n_neighbors=1, metric="correlation", n_jobs=-1),
        X, y_encoded, classes, CONFIG["n_iterations"], CONFIG["mask_fraction"],
        np.random.RandomState(seed)
    )
    all_baselines.append(result)
    print(f"   AUC={result['mean_auc']:.3f}, F1={result['mean_f1']:.3f}")

    # 4. k-NN (k=5)
    print("\n4. k-NN (k=5)...")
    result = run_baseline(
        "5-NN",
        lambda: KNeighborsClassifier(n_neighbors=5, metric="correlation",
                                      weights="distance", n_jobs=-1),
        X, y_encoded, classes, CONFIG["n_iterations"], CONFIG["mask_fraction"],
        np.random.RandomState(seed)
    )
    all_baselines.append(result)
    print(f"   AUC={result['mean_auc']:.3f}, F1={result['mean_f1']:.3f}")

    # Load label propagation results if available
    lp_report = Path(CONFIG["reports_dir"]) / "baseline_label_propagation.md"
    if lp_report.exists():
        print("\n5. Loading label propagation results...")
        # Parse AUC from report
        with open(lp_report) as f:
            content = f.read()
        import re
        auc_match = re.search(r"\*\*Mean AUC: ([\d.]+)\*\*", content)
        f1_match = re.search(r"\*\*Mean F1: ([\d.]+)\*\*", content)
        mcc_match = re.search(r"\*\*Mean MCC: ([\d.]+)\*\*", content)

        if auc_match:
            all_baselines.append({
                "name": "Label Propagation",
                "mean_auc": float(auc_match.group(1)),
                "std_auc": 0.0,
                "mean_f1": float(f1_match.group(1)) if f1_match else 0.0,
                "mean_mcc": float(mcc_match.group(1)) if mcc_match else 0.0,
            })
            print(f"   AUC={all_baselines[-1]['mean_auc']:.3f}")

    # --- Load XGBoost results if available ---
    xgb_report = Path(CONFIG["reports_dir"]) / "training_report.md"
    if xgb_report.exists():
        with open(xgb_report) as f:
            content = f.read()
        import re
        auc_match = re.search(r"\*\*Overall AUC\*\*: ([\d.]+)", content)
        if auc_match:
            # Extract F1 and MCC from Stage 1 (primary classification stage)
            f1_match = re.search(r"\*\*F1\*\*: ([\d.]+)", content)
            mcc_match = re.search(r"\*\*MCC\*\*: ([\d.]+)", content)
            all_baselines.append({
                "name": "XGBoost (two-stage)",
                "mean_auc": float(auc_match.group(1)),
                "std_auc": 0.0,
                "mean_f1": float(f1_match.group(1)) if f1_match else 0.0,
                "mean_mcc": float(mcc_match.group(1)) if mcc_match else 0.0,
            })

    # --- Summary table ---
    print(f"\n{'='*60}")
    print(f"  {'Model':<30} {'AUC':>8} {'F1':>8} {'MCC':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for b in sorted(all_baselines, key=lambda x: x["mean_auc"], reverse=True):
        print(f"  {b['name']:<30} {b['mean_auc']:>8.3f} {b['mean_f1']:>8.3f} {b['mean_mcc']:>8.3f}")
    print(f"{'='*60}")

    # --- Plot ---
    fig_dir = Path(CONFIG["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    names = [b["name"] for b in all_baselines]
    aucs = [b["mean_auc"] for b in all_baselines]
    f1s = [b["mean_f1"] for b in all_baselines]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, aucs, width, label="AUC", color="steelblue")
    ax.bar(x + width/2, f1s, width, label="F1", color="coral")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Baselines Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(fig_dir / "baselines_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Save report ---
    report_dir = Path(CONFIG["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baselines Comparison\n",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        "| Model | AUC | F1 | MCC |",
        "|-------|-----|----|----|",
    ]
    for b in sorted(all_baselines, key=lambda x: x["mean_auc"], reverse=True):
        lines.append(f"| {b['name']} | {b['mean_auc']:.3f} | {b['mean_f1']:.3f} | {b['mean_mcc']:.3f} |")

    lines.append("\n## Interpretation\n")
    lines.append("The XGBoost model must beat **all** baselines by >0.05 AUC to justify its complexity.")
    lines.append("If it cannot beat 1-NN or Label Propagation, the feature engineering adds no value.")

    with open(report_dir / "baselines_comparison.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Report: {report_dir / 'baselines_comparison.md'}")
    print(f"  Figure: {fig_dir / 'baselines_comparison.png'}")


if __name__ == "__main__":
    main()
