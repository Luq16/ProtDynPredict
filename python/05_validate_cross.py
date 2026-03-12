#!/usr/bin/env python3
"""
05_validate_cross.py - Cross-dataset validation across cancer types.

Validates ProtDynPredict transferability by training on one (or two)
cancer-type datasets and testing on the held-out cancer type.

Sections:
  1. Train-on-A, Test-on-B (all ordered pairs)
  2. Leave-One-Cancer-Out (train on 2, test on 1)
  3. Feature alignment report
  4. Per-class breakdown
  5. Confusion matrices

Input:  data/{dataset}/processed/feature_matrix_train.csv  (ucec, coad, brca)
Output: results/reports/cross_validation_report.md
        results/figures/cross_dataset_auc.png
        results/figures/cross_dataset_confusion.png
"""

import argparse
import sys
import warnings
from itertools import combinations, permutations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------ #
#  Constants & Configuration                                          #
# ------------------------------------------------------------------ #

DATASETS = ["ucec", "coad", "luad"]

LEAKY_PREFIXES = [
    "ppi_frac_neighbors_", "ppi_weighted_frac_",
    "pw_max_frac_", "pw_mean_frac_",
    "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_",
]

META_COLS = ["UniProt_ID", "label", "log2FC", "adj_pvalue"]

CONFIG = {
    "data_root": "data",
    "model_dir": "models",
    "figures_dir": "results/figures",
    "reports_dir": "results/reports",
    "random_state": 42,
}

DEFAULT_XGB_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


# ------------------------------------------------------------------ #
#  Data Loading                                                       #
# ------------------------------------------------------------------ #

def _is_leaky(col_name: str) -> bool:
    return any(col_name.startswith(p) for p in LEAKY_PREFIXES)


def _feature_path(dataset: str) -> Path:
    """Return the expected path for a dataset's feature matrix."""
    return Path(CONFIG["data_root"]) / dataset / "processed" / "feature_matrix_train.csv"


def load_dataset(dataset_name: str) -> dict | None:
    """Load a single dataset and return a dict with X, labels, feature_cols, df.

    Returns None (with a warning) if the file does not exist.
    """
    path = _feature_path(dataset_name)
    if not path.exists():
        print(f"  WARNING: Dataset '{dataset_name}' not found at {path} -- skipping.")
        return None

    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns
                    if c not in META_COLS and not _is_leaky(c)]

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    labels = df["label"].values  # up / down / unchanged

    print(f"  Loaded {dataset_name}: {X.shape[0]} proteins, "
          f"{X.shape[1]} features, classes={dict(zip(*np.unique(labels, return_counts=True)))}")

    return {
        "name": dataset_name,
        "df": df,
        "X": X,
        "labels": labels,
        "feature_cols": feature_cols,
    }


def load_all_datasets() -> dict:
    """Load all available datasets. Returns {name: dataset_dict}."""
    loaded = {}
    for ds in DATASETS:
        result = load_dataset(ds)
        if result is not None:
            loaded[ds] = result
    return loaded


# ------------------------------------------------------------------ #
#  Feature Alignment                                                  #
# ------------------------------------------------------------------ #

def align_features(datasets: dict) -> tuple[list[str], dict]:
    """Align features across datasets (intersection of safe columns).

    Returns:
        shared_cols: list of feature column names common to all datasets
        alignment_info: dict with shared / unique-per-dataset info
    """
    all_col_sets = {name: set(ds["feature_cols"]) for name, ds in datasets.items()}

    shared = set.intersection(*all_col_sets.values())
    shared_cols = sorted(shared)  # deterministic ordering

    alignment_info = {
        "n_shared": len(shared_cols),
        "per_dataset": {},
    }
    for name, cols in all_col_sets.items():
        unique = cols - shared
        alignment_info["per_dataset"][name] = {
            "total": len(cols),
            "unique": sorted(unique),
            "n_unique": len(unique),
        }

    print(f"\n  Feature alignment: {len(shared_cols)} shared features")
    for name, info in alignment_info["per_dataset"].items():
        print(f"    {name}: {info['total']} total, {info['n_unique']} unique (not in other datasets)")

    return shared_cols, alignment_info


def _extract_aligned(dataset: dict, shared_cols: list[str]) -> np.ndarray:
    """Extract feature matrix for shared columns only."""
    col_idx = [dataset["feature_cols"].index(c) for c in shared_cols
               if c in dataset["feature_cols"]]
    return dataset["X"][:, col_idx]


# ------------------------------------------------------------------ #
#  Label Preparation                                                  #
# ------------------------------------------------------------------ #

def prepare_binary_labels(labels: np.ndarray, stage: int = 1):
    """Prepare binary labels for a given stage.

    Stage 1: DE (up|down)=1 vs unchanged=0
    Stage 2: up=1 vs down=0  (DE samples only)

    Returns (y_binary, mask) where mask indicates which samples are used.
    """
    if stage == 1:
        y = np.where(labels == "unchanged", 0, 1)
        mask = np.ones(len(labels), dtype=bool)
    else:
        mask = np.isin(labels, ["up", "down"])
        y = np.where(labels == "up", 1, 0)
    return y, mask


# ------------------------------------------------------------------ #
#  Model Training & Evaluation                                        #
# ------------------------------------------------------------------ #

def _load_stored_params() -> dict | None:
    """Try to load hyperparameters from the saved stage-1 model."""
    model_path = Path(CONFIG["model_dir"]) / "stage1_model.joblib"
    if not model_path.exists():
        return None
    try:
        bundle = joblib.load(model_path)
        params = bundle.get("params")
        if params:
            print("  Using stored hyperparameters from stage1_model.joblib")
            return params
    except Exception as exc:
        print(f"  WARNING: Could not load stored params: {exc}")
    return None


def get_xgb_params() -> dict:
    """Return XGBoost parameters: stored if available, else defaults."""
    stored = _load_stored_params()
    if stored is not None:
        # Ensure essential keys are present
        stored.setdefault("objective", "binary:logistic")
        stored.setdefault("eval_metric", "auc")
        stored.setdefault("n_jobs", -1)
        stored.setdefault("verbosity", 0)
        return stored
    print("  Using default XGBoost parameters")
    return dict(DEFAULT_XGB_PARAMS)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                         y_proba: np.ndarray) -> dict:
    """Compute AUC, F1, MCC for binary predictions."""
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    f1 = f1_score(y_true, y_pred, average="binary")
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return {"auc": auc, "f1": f1, "mcc": mcc, "cm": cm}


def train_and_evaluate(X_train, y_train, X_test, y_test, params):
    """Train XGBoost on (X_train, y_train), evaluate on (X_test, y_test)."""
    scale_pos = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
    run_params = dict(params)
    run_params["scale_pos_weight"] = scale_pos

    clf = xgb.XGBClassifier(**run_params)
    clf.fit(X_train, y_train, verbose=False)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(y_test, y_pred, y_proba)
    return metrics, clf


# ------------------------------------------------------------------ #
#  Section 1: Train-on-A, Test-on-B                                   #
# ------------------------------------------------------------------ #

def pairwise_cross_validation(datasets: dict, shared_cols: list[str],
                              params: dict) -> list[dict]:
    """Train on dataset A, test on dataset B for every ordered pair."""
    print("\n" + "=" * 60)
    print("  SECTION 1: Train-on-A, Test-on-B")
    print("=" * 60)

    results = []
    names = sorted(datasets.keys())

    for train_name, test_name in permutations(names, 2):
        for stage in [1, 2]:
            stage_label = "DE-vs-Unchanged" if stage == 1 else "Up-vs-Down"

            X_train_full = _extract_aligned(datasets[train_name], shared_cols)
            X_test_full = _extract_aligned(datasets[test_name], shared_cols)

            y_train_all, mask_train = prepare_binary_labels(
                datasets[train_name]["labels"], stage)
            y_test_all, mask_test = prepare_binary_labels(
                datasets[test_name]["labels"], stage)

            X_tr = X_train_full[mask_train]
            y_tr = y_train_all[mask_train]
            X_te = X_test_full[mask_test]
            y_te = y_test_all[mask_test]

            # Need both classes in train and test
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                print(f"  {train_name}->{test_name} [{stage_label}]: "
                      "SKIPPED (single class)")
                continue

            metrics, _ = train_and_evaluate(X_tr, y_tr, X_te, y_te, params)
            results.append({
                "train": train_name,
                "test": test_name,
                "stage": stage,
                "stage_label": stage_label,
                **metrics,
            })

            print(f"  {train_name} -> {test_name} [{stage_label}]: "
                  f"AUC={metrics['auc']:.3f}  F1={metrics['f1']:.3f}  "
                  f"MCC={metrics['mcc']:.3f}")

    return results


# ------------------------------------------------------------------ #
#  Section 2: Leave-One-Cancer-Out                                    #
# ------------------------------------------------------------------ #

def leave_one_cancer_out(datasets: dict, shared_cols: list[str],
                         params: dict) -> list[dict]:
    """Train on N-1 datasets, test on the held-out one."""
    print("\n" + "=" * 60)
    print("  SECTION 2: Leave-One-Cancer-Out")
    print("=" * 60)

    results = []
    names = sorted(datasets.keys())

    if len(names) < 2:
        print("  Need at least 2 datasets; skipping LOCO.")
        return results

    for held_out in names:
        train_names = [n for n in names if n != held_out]

        for stage in [1, 2]:
            stage_label = "DE-vs-Unchanged" if stage == 1 else "Up-vs-Down"

            # Concatenate training data
            X_parts, y_parts = [], []
            for tn in train_names:
                X_full = _extract_aligned(datasets[tn], shared_cols)
                y_all, mask = prepare_binary_labels(datasets[tn]["labels"], stage)
                X_parts.append(X_full[mask])
                y_parts.append(y_all[mask])

            X_tr = np.vstack(X_parts)
            y_tr = np.concatenate(y_parts)

            # Test data
            X_test_full = _extract_aligned(datasets[held_out], shared_cols)
            y_test_all, mask_test = prepare_binary_labels(
                datasets[held_out]["labels"], stage)
            X_te = X_test_full[mask_test]
            y_te = y_test_all[mask_test]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                print(f"  Train={'+'.join(train_names)}, Test={held_out} "
                      f"[{stage_label}]: SKIPPED (single class)")
                continue

            metrics, _ = train_and_evaluate(X_tr, y_tr, X_te, y_te, params)
            results.append({
                "train": "+".join(train_names),
                "test": held_out,
                "stage": stage,
                "stage_label": stage_label,
                **metrics,
            })

            print(f"  Train={'+'.join(train_names)}, Test={held_out} "
                  f"[{stage_label}]: AUC={metrics['auc']:.3f}  "
                  f"F1={metrics['f1']:.3f}  MCC={metrics['mcc']:.3f}")

    return results


# ------------------------------------------------------------------ #
#  Section 4: Per-Class Breakdown                                     #
# ------------------------------------------------------------------ #

def per_class_breakdown(datasets: dict, shared_cols: list[str],
                        params: dict) -> list[dict]:
    """Per-class AUC for cross-dataset tests (stage 1 only: DE vs unchanged).

    Reports AUC separately for up-regulated, down-regulated, and unchanged
    proteins to identify if specific directions transfer poorly.
    """
    print("\n" + "=" * 60)
    print("  SECTION 4: Per-Class Breakdown (Stage 1)")
    print("=" * 60)

    results = []
    names = sorted(datasets.keys())

    for train_name, test_name in permutations(names, 2):
        X_tr = _extract_aligned(datasets[train_name], shared_cols)
        X_te = _extract_aligned(datasets[test_name], shared_cols)

        y_tr, _ = prepare_binary_labels(datasets[train_name]["labels"], stage=1)
        y_te, _ = prepare_binary_labels(datasets[test_name]["labels"], stage=1)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        scale_pos = np.sum(y_tr == 0) / max(np.sum(y_tr == 1), 1)
        run_params = dict(params)
        run_params["scale_pos_weight"] = scale_pos

        clf = xgb.XGBClassifier(**run_params)
        clf.fit(X_tr, y_tr, verbose=False)
        y_proba = clf.predict_proba(X_te)[:, 1]

        test_labels = datasets[test_name]["labels"]
        row = {"train": train_name, "test": test_name}

        for cls_name in ["up", "down", "unchanged"]:
            cls_mask = test_labels == cls_name
            if cls_mask.sum() == 0:
                row[f"n_{cls_name}"] = 0
                row[f"mean_proba_{cls_name}"] = float("nan")
                continue

            # Mean predicted probability of being DE for each class
            row[f"n_{cls_name}"] = int(cls_mask.sum())
            row[f"mean_proba_{cls_name}"] = float(np.mean(y_proba[cls_mask]))

        # Per-class precision: what fraction of truly-DE proteins the model
        # identifies with proba > 0.5, broken down by original direction
        y_pred = (y_proba > 0.5).astype(int)
        for cls_name in ["up", "down"]:
            cls_mask = test_labels == cls_name
            if cls_mask.sum() > 0:
                row[f"recall_{cls_name}"] = float(np.mean(y_pred[cls_mask] == 1))
            else:
                row[f"recall_{cls_name}"] = float("nan")

        unch_mask = test_labels == "unchanged"
        if unch_mask.sum() > 0:
            row["specificity_unchanged"] = float(np.mean(y_pred[unch_mask] == 0))
        else:
            row["specificity_unchanged"] = float("nan")

        results.append(row)
        print(f"  {train_name} -> {test_name}: "
              f"recall_up={row.get('recall_up', 'N/A'):.3f}  "
              f"recall_down={row.get('recall_down', 'N/A'):.3f}  "
              f"spec_unch={row.get('specificity_unchanged', 'N/A'):.3f}")

    return results


# ------------------------------------------------------------------ #
#  Section 5: Confusion Matrix Visualization                          #
# ------------------------------------------------------------------ #

def plot_confusion_matrices(pairwise_results: list[dict],
                            output_path: Path) -> None:
    """Plot cross-dataset confusion matrices (stage 1 only) in a grid."""
    stage1 = [r for r in pairwise_results if r["stage"] == 1]
    if not stage1:
        print("  No stage-1 pairwise results to plot.")
        return

    n = len(stage1)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    class_labels = ["Unchanged", "DE"]

    for idx, res in enumerate(stage1):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        cm = res["cm"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{res['train']} -> {res['test']}\n"
                     f"AUC={res['auc']:.3f}", fontsize=10)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    plt.suptitle("Cross-Dataset Confusion Matrices (Stage 1: DE vs Unchanged)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ------------------------------------------------------------------ #
#  AUC Bar Chart                                                      #
# ------------------------------------------------------------------ #

def plot_auc_chart(pairwise_results: list[dict],
                   loco_results: list[dict],
                   output_path: Path) -> None:
    """Bar chart of AUC across all cross-dataset experiments."""
    all_results = []

    for r in pairwise_results:
        all_results.append({
            "experiment": f"{r['train']}->{r['test']}",
            "stage": r["stage_label"],
            "auc": r["auc"],
        })
    for r in loco_results:
        all_results.append({
            "experiment": f"LOCO: test={r['test']}",
            "stage": r["stage_label"],
            "auc": r["auc"],
        })

    if not all_results:
        print("  No results to plot for AUC chart.")
        return

    df = pd.DataFrame(all_results)

    fig, ax = plt.subplots(figsize=(max(8, len(all_results) * 0.8), 6))
    experiments = df["experiment"].unique()
    stages = df["stage"].unique()
    x = np.arange(len(experiments))
    width = 0.35

    for i, stage in enumerate(stages):
        subset = df[df["stage"] == stage]
        # Align bars with experiment positions
        aucs = []
        for exp in experiments:
            match = subset[subset["experiment"] == exp]
            aucs.append(match["auc"].values[0] if len(match) > 0 else 0)
        offset = (i - len(stages) / 2 + 0.5) * width
        bars = ax.bar(x + offset, aucs, width, label=stage)
        # Annotate
        for bar, val in zip(bars, aucs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.6, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("AUC")
    ax.set_title("Cross-Dataset Validation AUC")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.08)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ------------------------------------------------------------------ #
#  Report Generation                                                  #
# ------------------------------------------------------------------ #

def generate_report(pairwise: list[dict], loco: list[dict],
                    alignment: dict, per_class: list[dict],
                    datasets: dict, output_path: Path) -> None:
    """Write a comprehensive Markdown report."""
    lines = []
    lines.append("# ProtDynPredict -- Cross-Dataset Validation Report\n")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"**Datasets**: {', '.join(sorted(datasets.keys()))}\n")

    # --- Feature alignment ---
    lines.append("## Feature Alignment\n")
    lines.append(f"- **Shared features**: {alignment['n_shared']}")
    for name, info in alignment["per_dataset"].items():
        lines.append(f"- **{name}**: {info['total']} total, "
                      f"{info['n_unique']} dataset-specific")
    if any(info["n_unique"] > 0 for info in alignment["per_dataset"].values()):
        lines.append("\n### Dataset-Specific Features\n")
        for name, info in alignment["per_dataset"].items():
            if info["n_unique"] > 0:
                lines.append(f"**{name}** ({info['n_unique']}):")
                for col in info["unique"][:20]:
                    lines.append(f"  - `{col}`")
                if info["n_unique"] > 20:
                    lines.append(f"  - ... and {info['n_unique'] - 20} more")
    lines.append("")

    # --- Pairwise ---
    lines.append("## Section 1: Train-on-A, Test-on-B\n")
    if pairwise:
        lines.append("| Train | Test | Stage | AUC | F1 | MCC |")
        lines.append("|-------|------|-------|-----|----|----|")
        for r in pairwise:
            auc_str = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "N/A"
            lines.append(f"| {r['train']} | {r['test']} | {r['stage_label']} | "
                          f"{auc_str} | {r['f1']:.3f} | {r['mcc']:.3f} |")
    else:
        lines.append("*No pairwise results (insufficient datasets).*")
    lines.append("")

    # --- LOCO ---
    lines.append("## Section 2: Leave-One-Cancer-Out\n")
    if loco:
        lines.append("| Training Set | Held-Out | Stage | AUC | F1 | MCC |")
        lines.append("|-------------|----------|-------|-----|----|----|")
        for r in loco:
            auc_str = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "N/A"
            lines.append(f"| {r['train']} | {r['test']} | {r['stage_label']} | "
                          f"{auc_str} | {r['f1']:.3f} | {r['mcc']:.3f} |")
    else:
        lines.append("*No LOCO results (need >= 2 datasets).*")
    lines.append("")

    # --- Per-class ---
    lines.append("## Section 4: Per-Class Breakdown (Stage 1)\n")
    if per_class:
        lines.append("| Train | Test | Recall(up) | Recall(down) | Spec(unchanged) "
                      "| MeanP(up) | MeanP(down) | MeanP(unch) |")
        lines.append("|-------|------|-----------|-------------|----------------|"
                      "----------|------------|------------|")
        for r in per_class:
            def _fmt(v):
                return f"{v:.3f}" if not (isinstance(v, float) and np.isnan(v)) else "N/A"
            lines.append(
                f"| {r['train']} | {r['test']} "
                f"| {_fmt(r.get('recall_up', float('nan')))} "
                f"| {_fmt(r.get('recall_down', float('nan')))} "
                f"| {_fmt(r.get('specificity_unchanged', float('nan')))} "
                f"| {_fmt(r.get('mean_proba_up', float('nan')))} "
                f"| {_fmt(r.get('mean_proba_down', float('nan')))} "
                f"| {_fmt(r.get('mean_proba_unchanged', float('nan')))} |"
            )
    else:
        lines.append("*No per-class results.*")
    lines.append("")

    # --- Interpretation ---
    lines.append("## Interpretation\n")
    if pairwise:
        aucs = [r["auc"] for r in pairwise if not np.isnan(r["auc"])]
        if aucs:
            mean_auc = np.mean(aucs)
            lines.append(f"- **Mean pairwise AUC**: {mean_auc:.3f}")
            if mean_auc > 0.7:
                lines.append("- The model transfers reasonably well across cancer types.")
            elif mean_auc > 0.6:
                lines.append("- Moderate transferability; cancer-type-specific tuning recommended.")
            else:
                lines.append("- Poor transferability; features may be cancer-type-specific.")
    if loco:
        aucs = [r["auc"] for r in loco if not np.isnan(r["auc"])]
        if aucs:
            lines.append(f"- **Mean LOCO AUC**: {np.mean(aucs):.3f}")

    lines.append("\n## Figures\n")
    lines.append("- `cross_dataset_auc.png` -- AUC bar chart for all experiments")
    lines.append("- `cross_dataset_confusion.png` -- Confusion matrices (Stage 1)")

    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report_text)
    print(f"\n  Report saved: {output_path}")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-dataset validation for ProtDynPredict.")
    parser.add_argument("--data-root", default=CONFIG["data_root"],
                        help="Root directory containing dataset folders")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        help="Dataset names to include")
    parser.add_argument("--figures-dir", default=CONFIG["figures_dir"])
    parser.add_argument("--reports-dir", default=CONFIG["reports_dir"])
    return parser.parse_args()


def main():
    args = parse_args()
    CONFIG["data_root"] = args.data_root
    CONFIG["figures_dir"] = args.figures_dir
    CONFIG["reports_dir"] = args.reports_dir
    CONFIG["model_dir"] = f"models/{args.datasets[0]}" if args.datasets else CONFIG["model_dir"]
    CONFIG["figures_dir"] = f"results/cross_dataset/figures"
    CONFIG["reports_dir"] = f"results/cross_dataset/reports"

    print("=" * 60)
    print("  Phase 5: CROSS-DATASET VALIDATION")
    print("=" * 60)

    # ---- Load datasets ----
    print("\nLoading datasets...")
    datasets = {}
    for ds_name in args.datasets:
        result = load_dataset(ds_name)
        if result is not None:
            datasets[ds_name] = result

    if len(datasets) < 2:
        print(f"\n  ERROR: Need at least 2 datasets for cross-validation, "
              f"found {len(datasets)}: {list(datasets.keys())}")
        print("  Ensure feature matrices exist at data/<dataset>/processed/feature_matrix_train.csv")
        sys.exit(1)

    # ---- Feature alignment ----
    print("\nAligning features...")
    shared_cols, alignment_info = align_features(datasets)

    if len(shared_cols) == 0:
        print("  ERROR: No shared features across datasets.")
        sys.exit(1)

    # ---- Get model params ----
    params = get_xgb_params()

    # ---- Section 1: Pairwise ----
    pairwise_results = pairwise_cross_validation(datasets, shared_cols, params)

    # ---- Section 2: Leave-One-Cancer-Out ----
    loco_results = leave_one_cancer_out(datasets, shared_cols, params)

    # ---- Section 4: Per-class breakdown ----
    per_class_results = per_class_breakdown(datasets, shared_cols, params)

    # ---- Section 5: Confusion matrices ----
    print("\n" + "=" * 60)
    print("  SECTION 5: Visualization")
    print("=" * 60)

    fig_dir = Path(CONFIG["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrices(pairwise_results,
                            fig_dir / "cross_dataset_confusion.png")
    plot_auc_chart(pairwise_results, loco_results,
                   fig_dir / "cross_dataset_auc.png")

    # ---- Report ----
    generate_report(
        pairwise_results, loco_results, alignment_info,
        per_class_results, datasets,
        Path(CONFIG["reports_dir"]) / "cross_validation_report.md",
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  CROSS-DATASET VALIDATION COMPLETE")
    print("=" * 60)

    all_aucs = [r["auc"] for r in pairwise_results + loco_results
                if not np.isnan(r.get("auc", float("nan")))]
    if all_aucs:
        print(f"  Overall mean AUC: {np.mean(all_aucs):.3f} "
              f"(range {np.min(all_aucs):.3f} -- {np.max(all_aucs):.3f})")
    print(f"  Figures: {fig_dir}/")
    print(f"  Report:  {Path(CONFIG['reports_dir']) / 'cross_validation_report.md'}")


if __name__ == "__main__":
    main()
