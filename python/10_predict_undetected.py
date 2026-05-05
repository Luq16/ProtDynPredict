#!/usr/bin/env python3
"""
10_predict_undetected.py - Predict DE status for proteins undetected in one dataset.

Core validation experiment: trains on dataset A (all proteins), then predicts
DE status for proteins present in dataset B but ABSENT from A. These simulate
proteins that were "undetected" in A's proteomics experiment.

For each ordered pair (train, test):
  - Non-overlapping proteins in test = "undetected" set
  - Overlapping proteins in test = control set
  - Evaluate both with AUC, F1, MCC, precision, recall

Input:  data/{dataset}/processed/feature_matrix_train.csv
Output: results/ucec/reports/undetected_prediction_report.md
        results/ucec/figures/undetected_prediction_auc.png
        results/ucec/figures/undetected_prediction_calibration.png
"""

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

DATASETS = ["ucec", "coad", "luad"]

LEAKY_PREFIXES = [
    "ppi_frac_neighbors_", "ppi_weighted_frac_",
    "pw_max_frac_", "pw_mean_frac_",
    "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_",
]

META_COLS = ["UniProt_ID", "label", "log2FC", "adj_pvalue"]

DEFAULT_XGB_PARAMS = {
    "max_depth": 5, "learning_rate": 0.1, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "binary:logistic", "eval_metric": "auc",
    "random_state": 42, "n_jobs": -1, "verbosity": 0,
}


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def _is_leaky(col: str) -> bool:
    return any(col.startswith(p) for p in LEAKY_PREFIXES)


def load_dataset(name: str) -> pd.DataFrame | None:
    path = Path("data") / name / "processed" / "feature_matrix_train.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found -- skipping {name}")
        return None
    return pd.read_csv(path)


def safe_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and not _is_leaky(c)]


def get_xgb_params(dataset: str) -> dict:
    for model_path in [
        Path("models") / dataset / "stage1_improved_model.joblib",
        Path("models") / dataset / "stage1_model.joblib",
    ]:
        if model_path.exists():
            try:
                bundle = joblib.load(model_path)
                params = bundle.get("params")
                if params:
                    params.setdefault("objective", "binary:logistic")
                    params.setdefault("eval_metric", "auc")
                    params.setdefault("n_jobs", -1)
                    params.setdefault("verbosity", 0)
                    print(f"  Using stored params from {model_path}")
                    return params
            except Exception:
                pass
    print(f"  Using default XGBoost params for {dataset}")
    return dict(DEFAULT_XGB_PARAMS)


def binary_labels(labels: np.ndarray) -> np.ndarray:
    """Stage 1: DE=1, unchanged=0."""
    return np.where(labels == "unchanged", 0, 1)


def evaluate(y_true, y_pred, y_proba) -> dict:
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    return {
        "auc": auc,
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "n": len(y_true),
    }


# ------------------------------------------------------------------ #
#  Core experiment                                                    #
# ------------------------------------------------------------------ #

def run_pair(train_df: pd.DataFrame, test_df: pd.DataFrame,
             train_name: str, test_name: str, params: dict) -> dict:
    """Run undetected-protein prediction for one ordered pair."""

    train_ids = set(train_df["UniProt_ID"])
    test_ids = set(test_df["UniProt_ID"])

    overlap_ids = train_ids & test_ids
    undetected_ids = test_ids - train_ids  # in test but NOT in train

    print(f"\n  {train_name} -> {test_name}: "
          f"overlap={len(overlap_ids)}, undetected={len(undetected_ids)}")

    # Feature alignment: intersection of safe columns
    train_feats = set(safe_feature_cols(train_df))
    test_feats = set(safe_feature_cols(test_df))
    shared_cols = sorted(train_feats & test_feats)
    print(f"    Shared features: {len(shared_cols)}")

    # Prepare training data (ALL proteins in train_dataset)
    X_train = train_df[shared_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = binary_labels(train_df["label"].values)

    # Class balancing
    scale_pos = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
    run_params = dict(params)
    run_params["scale_pos_weight"] = scale_pos

    # Train model
    clf = xgb.XGBClassifier(**run_params)
    clf.fit(X_train, y_train, verbose=False)

    # Split test into undetected vs overlapping
    test_undetected = test_df[test_df["UniProt_ID"].isin(undetected_ids)]
    test_overlap = test_df[test_df["UniProt_ID"].isin(overlap_ids)]

    result = {
        "train": train_name, "test": test_name,
        "n_train": len(train_df), "n_overlap": len(test_overlap),
        "n_undetected": len(test_undetected), "n_shared_features": len(shared_cols),
    }

    for subset_name, subset_df in [("undetected", test_undetected),
                                    ("overlap", test_overlap)]:
        if len(subset_df) < 5:
            print(f"    {subset_name}: too few proteins ({len(subset_df)}), skipping")
            for k in ["auc", "f1", "mcc", "precision", "recall", "n"]:
                result[f"{subset_name}_{k}"] = float("nan")
            result[f"{subset_name}_y_proba"] = np.array([])
            result[f"{subset_name}_y_true"] = np.array([])
            result[f"{subset_name}_labels"] = np.array([])
            continue

        X_sub = subset_df[shared_cols].values.astype(np.float32)
        X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
        y_true = binary_labels(subset_df["label"].values)

        y_pred = clf.predict(X_sub)
        y_proba = clf.predict_proba(X_sub)[:, 1]

        metrics = evaluate(y_true, y_pred, y_proba)
        for k, v in metrics.items():
            result[f"{subset_name}_{k}"] = v

        # Store for calibration plot
        result[f"{subset_name}_y_proba"] = y_proba
        result[f"{subset_name}_y_true"] = y_true
        result[f"{subset_name}_labels"] = subset_df["label"].values

        # Per-class recall among undetected
        if subset_name == "undetected":
            labels_raw = subset_df["label"].values
            for cls in ["up", "down", "unchanged"]:
                mask = labels_raw == cls
                if mask.sum() > 0:
                    if cls in ("up", "down"):
                        result[f"undetected_recall_{cls}"] = float(np.mean(y_pred[mask] == 1))
                    else:
                        result[f"undetected_recall_{cls}"] = float(np.mean(y_pred[mask] == 0))
                    result[f"undetected_n_{cls}"] = int(mask.sum())
                else:
                    result[f"undetected_recall_{cls}"] = float("nan")
                    result[f"undetected_n_{cls}"] = 0

            # Precision (PPV) for predicted-DE among undetected
            predicted_de_mask = y_pred == 1
            n_predicted_de = int(predicted_de_mask.sum())
            if n_predicted_de > 0:
                ppv = float(np.mean(y_true[predicted_de_mask] == 1))
            else:
                ppv = float("nan")
            result["undetected_ppv"] = ppv
            result["undetected_n_predicted_de"] = n_predicted_de

            # DE base rate among undetected proteins in this pair
            de_base_rate = float(np.mean(y_true == 1))
            result["undetected_de_base_rate"] = de_base_rate

            # Lift: model recall / base-rate recall
            # base-rate recall = DE base rate (what you'd get predicting all as DE)
            # Actually, lift = PPV / base_rate
            if de_base_rate > 0 and not np.isnan(ppv):
                result["undetected_lift"] = ppv / de_base_rate
            else:
                result["undetected_lift"] = float("nan")

        print(f"    {subset_name}: AUC={metrics['auc']:.3f}  F1={metrics['f1']:.3f}  "
              f"MCC={metrics['mcc']:.3f}  n={metrics['n']}")

    return result


# ------------------------------------------------------------------ #
#  Visualization                                                      #
# ------------------------------------------------------------------ #

def plot_auc_comparison(results: list[dict], out_path: Path):
    """Bar chart: undetected vs overlap AUC per pair."""
    pairs = [f"{r['train']}->{r['test']}" for r in results]
    auc_undet = [r.get("undetected_auc", float("nan")) for r in results]
    auc_over = [r.get("overlap_auc", float("nan")) for r in results]

    x = np.arange(len(pairs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.5), 5))
    b1 = ax.bar(x - w / 2, auc_undet, w, label="Undetected", color="#e74c3c", alpha=0.85)
    b2 = ax.bar(x + w / 2, auc_over, w, label="Overlap (control)", color="#3498db", alpha=0.85)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="gray", ls=":", alpha=0.6, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUC (Stage 1: DE vs Unchanged)")
    ax.set_title("Undetected Protein Prediction: AUC by Dataset Pair")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_calibration(results: list[dict], out_path: Path):
    """Predicted probability distributions for true-DE vs true-unchanged among undetected."""
    valid = [r for r in results
             if len(r.get("undetected_y_proba", [])) > 0]
    if not valid:
        print("  No data for calibration plot.")
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, r in enumerate(valid):
        ax = axes[0, idx]
        y_proba = r["undetected_y_proba"]
        y_true = r["undetected_y_true"]

        de_proba = y_proba[y_true == 1]
        unch_proba = y_proba[y_true == 0]

        if len(de_proba) > 0:
            ax.hist(de_proba, bins=25, alpha=0.6, color="#e74c3c",
                    label=f"True DE (n={len(de_proba)})", density=True)
        if len(unch_proba) > 0:
            ax.hist(unch_proba, bins=25, alpha=0.6, color="#3498db",
                    label=f"True Unchanged (n={len(unch_proba)})", density=True)

        ax.axvline(0.5, color="gray", ls="--", alpha=0.5)
        ax.set_xlabel("Predicted P(DE)")
        ax.set_ylabel("Density")
        ax.set_title(f"{r['train']}->{r['test']}\nAUC={r.get('undetected_auc', 0):.3f}",
                     fontsize=10)
        ax.legend(fontsize=8)

    plt.suptitle("Probability Distributions for Undetected Proteins", fontsize=12, y=1.02)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ------------------------------------------------------------------ #
#  Report                                                             #
# ------------------------------------------------------------------ #

def generate_report(results: list[dict], out_path: Path):
    L = []
    L.append("# Undetected Protein Prediction Report\n")
    L.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    L.append("## Experiment Design\n")
    L.append("For each ordered pair (A, B), train on **all** proteins in A, then predict")
    L.append("DE status for proteins in B that are **absent from A** (simulating undetected")
    L.append("proteins). Overlapping proteins serve as a control.\n")

    # Summary table
    L.append("## Summary Table\n")
    L.append("| Train | Test | N_train | N_undetected | N_overlap | AUC_undet | AUC_overlap | F1_undet | MCC_undet |")
    L.append("|-------|------|---------|-------------|-----------|-----------|-------------|----------|-----------|")
    for r in results:
        def _f(v): return f"{v:.3f}" if not np.isnan(v) else "N/A"
        L.append(f"| {r['train']} | {r['test']} | {r['n_train']} | {r['n_undetected']} "
                 f"| {r['n_overlap']} | {_f(r.get('undetected_auc', float('nan')))} "
                 f"| {_f(r.get('overlap_auc', float('nan')))} "
                 f"| {_f(r.get('undetected_f1', float('nan')))} "
                 f"| {_f(r.get('undetected_mcc', float('nan')))} |")

    # Per-class breakdown for undetected
    L.append("\n## Per-Class Breakdown (Undetected Proteins)\n")
    L.append("| Train | Test | Recall(up) | Recall(down) | Specificity(unchanged) | N_up | N_down | N_unch |")
    L.append("|-------|------|-----------|-------------|----------------------|------|--------|--------|")
    for r in results:
        def _f(v): return f"{v:.3f}" if not np.isnan(v) else "N/A"
        L.append(f"| {r['train']} | {r['test']} "
                 f"| {_f(r.get('undetected_recall_up', float('nan')))} "
                 f"| {_f(r.get('undetected_recall_down', float('nan')))} "
                 f"| {_f(r.get('undetected_recall_unchanged', float('nan')))} "
                 f"| {r.get('undetected_n_up', 0)} "
                 f"| {r.get('undetected_n_down', 0)} "
                 f"| {r.get('undetected_n_unchanged', 0)} |")

    # Precision, base rate, and lift for undetected
    L.append("\n## Precision, Base Rate, and Lift (Undetected Proteins)\n")
    L.append("| Train | Test | PPV (Precision) | DE Base Rate | Lift | N_predicted_DE |")
    L.append("|-------|------|-----------------|-------------|------|----------------|")
    for r in results:
        def _f(v): return f"{v:.3f}" if not np.isnan(v) else "N/A"
        L.append(f"| {r['train']} | {r['test']} "
                 f"| {_f(r.get('undetected_ppv', float('nan')))} "
                 f"| {_f(r.get('undetected_de_base_rate', float('nan')))} "
                 f"| {_f(r.get('undetected_lift', float('nan')))} "
                 f"| {r.get('undetected_n_predicted_de', 0)} |")

    # Summarize recall ranges accurately
    recall_ups = [r.get('undetected_recall_up', float('nan')) for r in results]
    recall_ups = [v for v in recall_ups if not np.isnan(v)]
    recall_downs = [r.get('undetected_recall_down', float('nan')) for r in results]
    recall_downs = [v for v in recall_downs if not np.isnan(v)]
    ppvs = [r.get('undetected_ppv', float('nan')) for r in results]
    ppvs = [v for v in ppvs if not np.isnan(v)]
    lifts = [r.get('undetected_lift', float('nan')) for r in results]
    lifts = [v for v in lifts if not np.isnan(v)]

    L.append("\n### Summary Statistics\n")
    if recall_ups:
        L.append(f"- **Recall (upregulated)**: {np.min(recall_ups):.0%} -- {np.max(recall_ups):.0%} "
                 f"(mean {np.mean(recall_ups):.0%})")
    if recall_downs:
        L.append(f"- **Recall (downregulated)**: {np.min(recall_downs):.0%} -- {np.max(recall_downs):.0%} "
                 f"(mean {np.mean(recall_downs):.0%})")
    if ppvs:
        L.append(f"- **Precision (PPV)**: {np.min(ppvs):.0%} -- {np.max(ppvs):.0%} "
                 f"(mean {np.mean(ppvs):.0%})")
    if lifts:
        L.append(f"- **Lift over base rate**: {np.min(lifts):.2f}x -- {np.max(lifts):.2f}x "
                 f"(mean {np.mean(lifts):.2f}x)")

    # AUC comparison
    undet_aucs = [r.get("undetected_auc", float("nan")) for r in results]
    over_aucs = [r.get("overlap_auc", float("nan")) for r in results]
    valid_undet = [a for a in undet_aucs if not np.isnan(a)]
    valid_over = [a for a in over_aucs if not np.isnan(a)]

    L.append("\n## AUC Comparison\n")
    if valid_undet:
        L.append(f"- **Mean AUC (undetected)**: {np.mean(valid_undet):.3f} "
                 f"(range {np.min(valid_undet):.3f} -- {np.max(valid_undet):.3f})")
    if valid_over:
        L.append(f"- **Mean AUC (overlap/control)**: {np.mean(valid_over):.3f} "
                 f"(range {np.min(valid_over):.3f} -- {np.max(valid_over):.3f})")
    if valid_undet and valid_over:
        delta = np.mean(valid_undet) - np.mean(valid_over)
        L.append(f"- **Delta (undetected - overlap)**: {delta:+.3f}")

    L.append("\n## Interpretation\n")
    if valid_undet:
        mean_u = np.mean(valid_undet)
        if mean_u > 0.65:
            L.append("The model **successfully predicts DE status for undetected proteins** "
                     f"(mean AUC={mean_u:.3f}), validating the core premise that intrinsic "
                     "protein features carry predictive signal for expression dynamics even "
                     "for proteins not observed in the training proteomics experiment.")
        elif mean_u > 0.55:
            L.append(f"Moderate predictive signal (mean AUC={mean_u:.3f}) for undetected "
                     "proteins. The model captures some generalizable patterns, though "
                     "performance is reduced compared to overlapping proteins.")
        else:
            L.append(f"Limited predictive signal (mean AUC={mean_u:.3f}) for undetected "
                     "proteins. The model struggles to generalize to proteins not seen "
                     "in the training dataset.")
    if valid_undet and valid_over:
        delta = np.mean(valid_undet) - np.mean(valid_over)
        if abs(delta) < 0.03:
            L.append("\nThe AUC gap between undetected and overlapping proteins is small, "
                     "suggesting the model generalizes well to unseen proteins.")
        elif delta < -0.03:
            L.append(f"\nAs expected, undetected proteins show lower AUC ({delta:+.3f}), "
                     "consistent with a modest domain shift for proteins absent from "
                     "training data.")

    L.append("\n## Figures\n")
    L.append("- `undetected_prediction_auc.png` -- AUC comparison bar chart")
    L.append("- `undetected_prediction_calibration.png` -- Probability distributions")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(L))
    print(f"\n  Report saved: {out_path}")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Predict DE status for undetected proteins across datasets.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Dataset names (default: ucec coad luad)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Phase 10: UNDETECTED PROTEIN PREDICTION")
    print("=" * 60)

    # Load all datasets
    data = {}
    for ds in args.datasets:
        df = load_dataset(ds)
        if df is not None:
            print(f"  {ds}: {len(df)} proteins, "
                  f"labels={df['label'].value_counts().to_dict()}")
            data[ds] = df

    if len(data) < 2:
        print("ERROR: Need at least 2 datasets.")
        return

    # Run all ordered pairs
    results = []
    names = sorted(data.keys())
    for train_name in names:
        params = get_xgb_params(train_name)
        for test_name in names:
            if train_name == test_name:
                continue
            r = run_pair(data[train_name], data[test_name],
                         train_name, test_name, params)
            results.append(r)

    # Output paths
    fig_dir = Path("results/ucec/figures")
    rep_dir = Path("results/ucec/reports")

    # Figures
    plot_auc_comparison(results, fig_dir / "undetected_prediction_auc.png")
    plot_calibration(results, fig_dir / "undetected_prediction_calibration.png")

    # Report
    generate_report(results, rep_dir / "undetected_prediction_report.md")

    # Final summary
    print("\n" + "=" * 60)
    print("  UNDETECTED PREDICTION COMPLETE")
    print("=" * 60)
    valid_u = [r.get("undetected_auc", float("nan")) for r in results]
    valid_u = [a for a in valid_u if not np.isnan(a)]
    valid_o = [r.get("overlap_auc", float("nan")) for r in results]
    valid_o = [a for a in valid_o if not np.isnan(a)]
    if valid_u:
        print(f"  Mean AUC (undetected):  {np.mean(valid_u):.3f}")
    if valid_o:
        print(f"  Mean AUC (overlap):     {np.mean(valid_o):.3f}")
    print(f"  Report:  {rep_dir / 'undetected_prediction_report.md'}")
    print(f"  Figures: {fig_dir}/")


if __name__ == "__main__":
    main()
