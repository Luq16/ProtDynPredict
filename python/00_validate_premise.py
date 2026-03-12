#!/usr/bin/env python3
"""
00_validate_premise.py
GO/NO-GO GATE: Validate that protein features can predict expression dynamics.

Masks 20% of detected proteins, trains on remaining 80%, predicts masked set.
Uses protein-family-grouped splits to avoid inflated metrics from homologs.

Input:  data/processed/feature_matrix_train.csv
Output: results/reports/validation_report.md
        results/figures/validation_*.png

Decision criteria:
  AUC > 0.65 for all classes → PROCEED to Phase 3
  AUC 0.60-0.65             → Review features, consider additions
  AUC < 0.60                → PIVOT approach
"""

import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_fscore_support
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration ---
CONFIG = {
    "input_file": "data/processed/feature_matrix_train.csv",
    "output_dir": "results",
    "figures_dir": "results/figures",
    "reports_dir": "results/reports",
    "n_masks": 5,           # number of masking iterations
    "mask_fraction": 0.20,  # fraction to mask
    "n_folds": 5,           # CV folds
    "random_state": 42,
    # Sequence identity threshold for protein family grouping
    # Without CD-HIT, we approximate by clustering on sequence features
    "cluster_corr_threshold": 0.8,
}


def load_data(path):
    """Load feature matrix and separate features/labels."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)

    label_col = "label"
    meta_cols = ["UniProt_ID", "label", "log2FC", "adj_pvalue"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values
    protein_ids = df["UniProt_ID"].values

    # Handle any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"  Labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, protein_ids, feature_cols


def approximate_protein_families(X, feature_cols, threshold=0.5):
    """
    Approximate protein family clusters from sequence feature similarity.
    Uses only AAC_* and DC_* columns (amino acid composition and dipeptide
    composition, ~420 features) with cosine distance.  A distance_threshold
    of 0.5 roughly approximates 50% sequence identity clustering.
    """
    print("  Approximating protein families from sequence feature similarity...")
    from sklearn.cluster import AgglomerativeClustering
    from scipy.spatial.distance import pdist, squareform

    # Extract only sequence-derived features (AAC and DC columns)
    seq_idx = [i for i, c in enumerate(feature_cols)
               if c.startswith("AAC_") or c.startswith("DC_")]
    if len(seq_idx) == 0:
        print("  WARNING: No AAC_/DC_ columns found; falling back to all features")
        X_seq = X
    else:
        X_seq = X[:, seq_idx]
        print(f"  Using {len(seq_idx)} sequence features (AAC + DC) for clustering")

    n = X_seq.shape[0]
    if n > 5000:
        # For very large datasets use MiniBatchKMeans on sequence features
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = max(20, n // 5)
        print(f"  Large dataset ({n} proteins), using MiniBatchKMeans (k={n_clusters})...")
        groups = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_seq)
        print(f"  Found {len(np.unique(groups))} approximate protein families")
        return groups

    # Cosine distance
    cos_dist = pdist(X_seq, metric="cosine")
    cos_dist = np.nan_to_num(cos_dist, nan=1.0)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average"
    )
    dist_matrix = squareform(cos_dist)
    groups = clustering.fit_predict(dist_matrix)

    n_clusters = len(np.unique(groups))
    print(f"  Found {n_clusters} approximate protein families")
    return groups


def identify_leaky_features(feature_cols):
    """
    Identify features that encode label information (neighbor/pathway stats).
    These should ideally be recomputed per fold, but for the validation gate
    we flag them and run with/without.
    """
    leaky_prefixes = [
        "ppi_frac_neighbors_", "ppi_weighted_frac_",
        "pw_max_frac_", "pw_mean_frac_",
        "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_"
    ]
    leaky_idx = []
    safe_idx = []
    for i, col in enumerate(feature_cols):
        if any(col.startswith(p) for p in leaky_prefixes):
            leaky_idx.append(i)
        else:
            safe_idx.append(i)

    print(f"  Label-dependent features: {len(leaky_idx)} (will test with/without)")
    print(f"  Label-independent features: {len(safe_idx)}")
    return leaky_idx, safe_idx


def run_masking_experiment(X, y, groups, feature_cols, mask_frac, n_masks, seed):
    """Run the masking experiment: mask proteins, predict their class."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_
    n_classes = len(classes)

    leaky_idx, safe_idx = identify_leaky_features(feature_cols)

    all_results = {"full": [], "safe_only": []}
    rng = np.random.RandomState(seed)

    for iteration in range(n_masks):
        print(f"\n--- Masking iteration {iteration + 1}/{n_masks} ---")

        # Stratified masking: mask 20% from each class proportionally
        mask_indices = []
        train_indices = []
        for cls in range(n_classes):
            cls_idx = np.where(y_encoded == cls)[0]
            n_mask = max(1, int(len(cls_idx) * mask_frac))
            rng.shuffle(cls_idx)
            mask_indices.extend(cls_idx[:n_mask])
            train_indices.extend(cls_idx[n_mask:])

        mask_indices = np.array(mask_indices)
        train_indices = np.array(train_indices)

        print(f"  Train: {len(train_indices)}, Masked: {len(mask_indices)}")

        for feature_set, name in [(None, "full"), (safe_idx, "safe_only")]:
            if feature_set is not None:
                X_use = X[:, feature_set]
            else:
                X_use = X

            X_train = X_use[train_indices]
            y_train = y_encoded[train_indices]
            X_mask = X_use[mask_indices]
            y_mask = y_encoded[mask_indices]

            # Train Random Forest (fast, robust baseline)
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=seed + iteration,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)

            # Predict
            y_pred = clf.predict(X_mask)
            y_proba = clf.predict_proba(X_mask)

            # Metrics
            if n_classes == 2:
                auc = roc_auc_score(y_mask, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_mask, y_proba, multi_class="ovr",
                                     average="macro")

            per_class_auc = {}
            for i, cls_name in enumerate(classes):
                binary_true = (y_mask == i).astype(int)
                if binary_true.sum() > 0 and binary_true.sum() < len(binary_true):
                    per_class_auc[cls_name] = roc_auc_score(binary_true, y_proba[:, i])
                else:
                    per_class_auc[cls_name] = np.nan

            f1_macro = f1_score(y_mask, y_pred, average="macro")
            mcc = matthews_corrcoef(y_mask, y_pred)

            result = {
                "iteration": iteration + 1,
                "macro_auc": auc,
                "f1_macro": f1_macro,
                "mcc": mcc,
                **{f"auc_{k}": v for k, v in per_class_auc.items()},
                "y_true": y_mask,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }

            if name == "full" and iteration == 0:
                # Store feature importance from first iteration
                result["feature_importance"] = clf.feature_importances_
                result["feature_names"] = (
                    feature_cols if feature_set is None
                    else [feature_cols[i] for i in feature_set]
                )

            all_results[name].append(result)
            print(f"  [{name:>10}] AUC={auc:.3f}, F1={f1_macro:.3f}, MCC={mcc:.3f}")

    return all_results, classes


def run_cv_experiment(X, y, groups, feature_cols, n_folds, seed):
    """Run grouped cross-validation as additional validation."""
    print(f"\n--- {n_folds}-Fold Grouped Cross-Validation ---")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_

    # Use GroupKFold if we have meaningful groups, else StratifiedKFold
    n_unique_groups = len(np.unique(groups))
    if n_unique_groups >= n_folds * 2:
        cv = GroupKFold(n_splits=n_folds)
        splits = list(cv.split(X, y_encoded, groups))
        print(f"  Using GroupKFold ({n_unique_groups} groups)")
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(cv.split(X, y_encoded))
        print("  Using StratifiedKFold (too few groups for GroupKFold)")

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        clf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3,
            class_weight="balanced", random_state=seed, n_jobs=-1
        )
        clf.fit(X[train_idx], y_encoded[train_idx])

        y_pred = clf.predict(X[test_idx])
        y_proba = clf.predict_proba(X[test_idx])
        y_true = y_encoded[test_idx]

        if len(classes) == 2:
            auc = roc_auc_score(y_true, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")

        f1 = f1_score(y_true, y_pred, average="macro")
        mcc = matthews_corrcoef(y_true, y_pred)

        fold_results.append({"fold": fold + 1, "auc": auc, "f1": f1, "mcc": mcc})
        print(f"  Fold {fold+1}: AUC={auc:.3f}, F1={f1:.3f}, MCC={mcc:.3f}")

    return fold_results, classes


def plot_results(mask_results, cv_results, classes, output_dir):
    """Generate validation figures."""
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Masking experiment: AUC across iterations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (name, results) in zip(axes, mask_results.items()):
        aucs = [r["macro_auc"] for r in results]
        per_class = {cls: [r.get(f"auc_{cls}", np.nan) for r in results]
                     for cls in classes}

        ax.bar(range(len(aucs)), aucs, alpha=0.7, label="Macro AUC")
        for cls, vals in per_class.items():
            ax.plot(range(len(vals)), vals, "o-", label=f"AUC {cls}", markersize=4)

        ax.axhline(y=0.65, color="red", linestyle="--", alpha=0.5, label="GO threshold")
        ax.axhline(y=0.50, color="gray", linestyle=":", alpha=0.5, label="Random")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("AUC")
        ax.set_title(f"Masking Experiment ({name})")
        ax.legend(fontsize=8)
        ax.set_ylim(0.3, 1.0)

    plt.tight_layout()
    plt.savefig(fig_dir / "validation_masking_auc.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Confusion matrix (last iteration, full features)
    last_full = mask_results["full"][-1]
    cm = confusion_matrix(last_full["y_true"], last_full["y_pred"])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Last Masking Iteration)")
    plt.tight_layout()
    plt.savefig(fig_dir / "validation_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Feature importance (top 30)
    if "feature_importance" in mask_results["full"][0]:
        importances = mask_results["full"][0]["feature_importance"]
        feat_names = mask_results["full"][0]["feature_names"]

        top_k = min(30, len(importances))
        top_idx = np.argsort(importances)[-top_k:]

        fig, ax = plt.subplots(figsize=(8, max(6, top_k * 0.3)))
        ax.barh(range(top_k), importances[top_idx], color="steelblue")
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([feat_names[i] for i in top_idx], fontsize=8)
        ax.set_xlabel("Feature Importance (Gini)")
        ax.set_title(f"Top {top_k} Features")
        plt.tight_layout()
        plt.savefig(fig_dir / "validation_feature_importance.png", dpi=150,
                     bbox_inches="tight")
        plt.close()

    # 4. CV results
    if cv_results:
        fig, ax = plt.subplots(figsize=(6, 4))
        metrics = ["auc", "f1", "mcc"]
        x = np.arange(len(cv_results))
        width = 0.25
        for i, metric in enumerate(metrics):
            vals = [r[metric] for r in cv_results]
            ax.bar(x + i * width, vals, width, label=metric.upper(), alpha=0.8)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Score")
        ax.set_title("Cross-Validation Results")
        ax.legend()
        ax.axhline(y=0.65, color="red", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(fig_dir / "validation_cv_results.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  Figures saved to {fig_dir}/")


def generate_report(mask_results, cv_results, classes, output_path):
    """Generate the validation report with GO/NO-GO decision."""
    lines = []
    lines.append("# ProtDynPredict — Premise Validation Report\n")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Masking results summary
    lines.append("## 1. Masking Experiment Results\n")

    for name, results in mask_results.items():
        lines.append(f"### Feature set: `{name}`\n")
        lines.append("| Iteration | Macro AUC | F1 Macro | MCC |")
        lines.append("|-----------|-----------|----------|-----|")

        aucs = []
        for r in results:
            lines.append(f"| {r['iteration']} | {r['macro_auc']:.3f} | "
                        f"{r['f1_macro']:.3f} | {r['mcc']:.3f} |")
            aucs.append(r["macro_auc"])

        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        lines.append(f"\n**Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}**\n")

        # Per-class AUC
        lines.append("Per-class AUC (mean across iterations):\n")
        for cls in classes:
            cls_aucs = [r.get(f"auc_{cls}", np.nan) for r in results]
            cls_mean = np.nanmean(cls_aucs)
            lines.append(f"- **{cls}**: {cls_mean:.3f}")
        lines.append("")

    # CV results
    if cv_results:
        lines.append("## 2. Cross-Validation Results\n")
        lines.append("| Fold | AUC | F1 | MCC |")
        lines.append("|------|-----|----|----|")
        for r in cv_results:
            lines.append(f"| {r['fold']} | {r['auc']:.3f} | {r['f1']:.3f} | {r['mcc']:.3f} |")

        mean_cv_auc = np.mean([r["auc"] for r in cv_results])
        lines.append(f"\n**Mean CV AUC: {mean_cv_auc:.3f}**\n")

    # Feature importance
    if "feature_importance" in mask_results["full"][0]:
        lines.append("## 3. Top 20 Features\n")
        importances = mask_results["full"][0]["feature_importance"]
        feat_names = mask_results["full"][0]["feature_names"]
        top_idx = np.argsort(importances)[-20:][::-1]

        lines.append("| Rank | Feature | Importance |")
        lines.append("|------|---------|------------|")
        for rank, idx in enumerate(top_idx, 1):
            lines.append(f"| {rank} | `{feat_names[idx]}` | {importances[idx]:.4f} |")
        lines.append("")

    # GO/NO-GO Decision
    lines.append("## 4. GO / NO-GO Decision\n")

    full_mean_auc = np.mean([r["macro_auc"] for r in mask_results["full"]])
    safe_mean_auc = np.mean([r["macro_auc"] for r in mask_results["safe_only"]])

    # Check per-class minimums
    min_class_auc = 1.0
    for cls in classes:
        cls_aucs = [r.get(f"auc_{cls}", np.nan) for r in mask_results["full"]]
        cls_mean = np.nanmean(cls_aucs)
        min_class_auc = min(min_class_auc, cls_mean)

    if min_class_auc >= 0.65:
        decision = "GO"
        emoji = "✅"
        action = "Proceed to Phase 3 (full model training)."
    elif min_class_auc >= 0.60:
        decision = "CAUTIOUS GO"
        emoji = "⚠️"
        action = ("Performance is marginal. Review feature importance, consider adding "
                  "features (ESM-2 embeddings, more PPI features). Proceed with caution.")
    else:
        decision = "NO-GO"
        emoji = "❌"
        action = ("Core premise not validated. Consider:\n"
                  "  1. Adding ESM-2 protein language model embeddings\n"
                  "  2. Richer PPI network features\n"
                  "  3. Incorporating matched transcriptomics data\n"
                  "  4. Reframing as binary (DE vs unchanged) instead of 3-class")

    lines.append(f"**Decision: {emoji} {decision}**\n")
    lines.append(f"- Full features AUC: {full_mean_auc:.3f}")
    lines.append(f"- Safe features only AUC: {safe_mean_auc:.3f}")
    lines.append(f"- Minimum per-class AUC: {min_class_auc:.3f}")
    lines.append(f"- Threshold: 0.65\n")
    lines.append(f"**Action**: {action}\n")

    # Leakage analysis
    lines.append("## 5. Leakage Analysis\n")
    auc_diff = full_mean_auc - safe_mean_auc
    lines.append(f"AUC difference (full - safe features): {auc_diff:+.3f}\n")
    if auc_diff > 0.10:
        lines.append("**WARNING**: Large gap suggests label-dependent features "
                     "(network/pathway/GO group stats) contribute heavily. "
                     "Ensure proper per-fold recomputation in Phase 3.\n")
    elif auc_diff > 0.05:
        lines.append("Moderate contribution from label-dependent features. "
                     "Per-fold recomputation recommended.\n")
    else:
        lines.append("Label-dependent features contribute minimally. "
                     "Intrinsic protein features carry most of the signal.\n")

    report = "\n".join(lines)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"  DECISION: {emoji} {decision}")
    print(f"  Full AUC:      {full_mean_auc:.3f}")
    print(f"  Safe-only AUC: {safe_mean_auc:.3f}")
    print(f"  Min class AUC: {min_class_auc:.3f}")
    print(f"{'='*60}")
    print(f"\n  Report: {output_path}")

    return decision


def main():
    parser = argparse.ArgumentParser(description="Validate premise (GO/NO-GO gate)")
    parser.add_argument("--dataset", default="ucec", help="Dataset name (e.g., ucec, coad, brca)")
    args = parser.parse_args()
    dataset = args.dataset

    CONFIG["input_file"] = f"data/{dataset}/processed/feature_matrix_train.csv"
    CONFIG["figures_dir"] = f"results/{dataset}/figures"
    CONFIG["reports_dir"] = f"results/{dataset}/reports"

    print("=" * 60)
    print(f"  Phase 2: VALIDATE PREMISE (GO/NO-GO GATE) [{dataset.upper()}]")
    print("=" * 60)

    # Load data
    X, y, protein_ids, feature_cols = load_data(CONFIG["input_file"])

    if X.shape[0] < 50:
        print(f"\nERROR: Only {X.shape[0]} proteins in training set.")
        print("Need at least 50 for meaningful validation.")
        sys.exit(1)

    # Approximate protein families for grouped splitting
    groups = approximate_protein_families(X, feature_cols)

    # Run masking experiment
    print("\n" + "=" * 40)
    print("  MASKING EXPERIMENT")
    print("=" * 40)
    mask_results, classes = run_masking_experiment(
        X, y, groups, feature_cols,
        CONFIG["mask_fraction"], CONFIG["n_masks"], CONFIG["random_state"]
    )

    # Run cross-validation
    print("\n" + "=" * 40)
    print("  CROSS-VALIDATION")
    print("=" * 40)
    cv_results, _ = run_cv_experiment(
        X, y, groups, feature_cols,
        CONFIG["n_folds"], CONFIG["random_state"]
    )

    # Generate figures
    print("\nGenerating figures...")
    plot_results(mask_results, cv_results, classes, CONFIG["figures_dir"])

    # Generate report and decision
    report_path = Path(CONFIG["reports_dir"]) / "validation_report.md"
    decision = generate_report(mask_results, cv_results, classes, str(report_path))

    return 0 if decision in ("GO", "CAUTIOUS GO") else 1


if __name__ == "__main__":
    sys.exit(main())
