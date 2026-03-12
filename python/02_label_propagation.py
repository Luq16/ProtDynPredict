#!/usr/bin/env python3
"""
02_label_propagation.py
Baseline: Label propagation on PPI network.

If XGBoost can't beat this, the ML model adds no value beyond
simple network diffusion.

Input:  data/processed/feature_matrix_train.csv
        data/processed/ppi_network.rds (loaded via network_features.csv)
Output: results/reports/baseline_label_propagation.md
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef, classification_report
)
from sklearn.preprocessing import LabelEncoder


def build_similarity_from_features(X, k=10):
    """Build k-NN similarity graph from features (proxy for PPI when full
    network adjacency matrix is not available in CSV format)."""
    from sklearn.metrics.pairwise import rbf_kernel
    from sklearn.neighbors import kneighbors_graph

    # k-NN graph with RBF weights
    knn = kneighbors_graph(X, n_neighbors=k, mode="distance", metric="euclidean")
    # Convert distance to similarity
    gamma = 1.0 / X.shape[1]
    knn.data = np.exp(-gamma * knn.data ** 2)

    return knn


def main():
    parser = argparse.ArgumentParser(description="Label propagation baseline")
    parser.add_argument("--dataset", default="ucec", help="Dataset/cancer type (default: ucec)")
    args = parser.parse_args()
    dataset = args.dataset

    CONFIG = {
        "train_file": f"data/{dataset}/processed/feature_matrix_train.csv",
        "network_file": f"data/{dataset}/processed/network_features.csv",
        "reports_dir": f"results/{dataset}/reports",
    }

    print("=" * 60)
    print("  BASELINE: Label Propagation")
    print(f"  Dataset: {dataset}")
    print("=" * 60)

    # Load data
    df = pd.read_csv(CONFIG["train_file"])
    meta_cols = ["UniProt_ID", "label", "log2FC", "adj_pvalue"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_

    print(f"  Proteins: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Classes: {dict(zip(classes, np.bincount(y_encoded)))}")

    # Leave-20%-out evaluation (same as masking experiment)
    n_iterations = 5
    mask_fraction = 0.20
    rng = np.random.RandomState(42)

    results = []

    for iteration in range(n_iterations):
        print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

        # Stratified masking
        mask_idx = []
        train_idx = []
        for cls in range(len(classes)):
            cls_indices = np.where(y_encoded == cls)[0]
            n_mask = max(1, int(len(cls_indices) * mask_fraction))
            rng.shuffle(cls_indices)
            mask_idx.extend(cls_indices[:n_mask])
            train_idx.extend(cls_indices[n_mask:])

        mask_idx = np.array(mask_idx)
        train_idx = np.array(train_idx)

        # Create label array: -1 for masked (unlabeled)
        y_semi = np.full(len(y_encoded), -1)
        y_semi[train_idx] = y_encoded[train_idx]

        # Label Spreading (more robust than Label Propagation)
        try:
            model = LabelSpreading(
                kernel="knn", n_neighbors=7, alpha=0.2, max_iter=100
            )
            model.fit(X, y_semi)

            y_pred = model.predict(X[mask_idx])
            y_proba = model.predict_proba(X[mask_idx])
            y_true = y_encoded[mask_idx]

            if len(classes) == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

            f1 = f1_score(y_true, y_pred, average="macro")
            mcc = matthews_corrcoef(y_true, y_pred)

            results.append({"iteration": iteration + 1, "auc": auc, "f1": f1, "mcc": mcc})
            print(f"  Label Spreading: AUC={auc:.3f}, F1={f1:.3f}, MCC={mcc:.3f}")

        except Exception as e:
            print(f"  Label Spreading failed: {e}")
            results.append({"iteration": iteration + 1, "auc": 0.5, "f1": 0.0, "mcc": 0.0})

    # Summary
    mean_auc = np.mean([r["auc"] for r in results])
    mean_f1 = np.mean([r["f1"] for r in results])
    mean_mcc = np.mean([r["mcc"] for r in results])

    print(f"\n{'='*60}")
    print(f"  Label Propagation Baseline")
    print(f"  Mean AUC: {mean_auc:.3f}")
    print(f"  Mean F1:  {mean_f1:.3f}")
    print(f"  Mean MCC: {mean_mcc:.3f}")
    print(f"{'='*60}")

    # Save report
    report_dir = Path(CONFIG["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline: Label Propagation\n",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Method",
        "Label Spreading (sklearn) with KNN kernel (k=7, alpha=0.2).",
        "20% of proteins masked per iteration, 5 iterations.\n",
        "## Results\n",
        "| Iteration | AUC | F1 | MCC |",
        "|-----------|-----|----|----|",
    ]
    for r in results:
        lines.append(f"| {r['iteration']} | {r['auc']:.3f} | {r['f1']:.3f} | {r['mcc']:.3f} |")
    lines.append(f"\n**Mean AUC: {mean_auc:.3f}**")
    lines.append(f"**Mean F1: {mean_f1:.3f}**")
    lines.append(f"**Mean MCC: {mean_mcc:.3f}**\n")
    lines.append("## Interpretation")
    lines.append("If XGBoost AUC is not substantially higher (>0.05) than this baseline,")
    lines.append("the ML model is essentially doing sophisticated label propagation")
    lines.append("and the extra feature engineering adds minimal value.")

    with open(report_dir / "baseline_label_propagation.md", "w") as f:
        f.write("\n".join(lines))

    print(f"  Report: {report_dir / 'baseline_label_propagation.md'}")


if __name__ == "__main__":
    main()
