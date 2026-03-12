#!/usr/bin/env python3
"""
01_train_model.py
Two-stage XGBoost model for predicting protein expression dynamics.

Stage 1: Binary — "differentially expressed" vs "unchanged"
Stage 2: Binary — "upregulated" vs "downregulated" (DE proteins only)

Uses protein-family-grouped CV and Optuna hyperparameter tuning.

Input:  data/processed/feature_matrix_train.csv
Output: models/stage1_model.joblib
        models/stage2_model.joblib
        results/reports/training_report.md
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Configuration ---
CONFIG = {
    "input_file": "data/processed/feature_matrix_train.csv",
    "model_dir": "models",
    "results_dir": "results",
    "figures_dir": "results/figures",
    "reports_dir": "results/reports",
    "n_folds": 5,
    "optuna_trials": 50,
    "random_state": 42,
}


def load_and_prepare(path):
    """Load data and prepare two-stage labels."""
    df = pd.read_csv(path)

    meta_cols = ["UniProt_ID", "label", "log2FC", "adj_pvalue"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Identify and exclude label-dependent features to prevent leakage
    # These encode neighbor/pathway expression stats computed from ALL labels
    leaky_prefixes = [
        "ppi_frac_neighbors_", "ppi_weighted_frac_",
        "pw_max_frac_", "pw_mean_frac_",
        "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_"
    ]
    safe_feature_cols = [c for c in feature_cols
                         if not any(c.startswith(p) for p in leaky_prefixes)]
    leaky_cols = [c for c in feature_cols if c not in safe_feature_cols]

    print(f"  Excluding {len(leaky_cols)} label-dependent features to prevent leakage")
    print(f"  Using {len(safe_feature_cols)} safe features")
    feature_cols = safe_feature_cols

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Stage 1 labels: DE (up or down) vs unchanged
    y_stage1 = np.where(df["label"] == "unchanged", 0, 1)  # 0=unchanged, 1=DE

    # Stage 2 labels: up vs down (only for DE proteins)
    de_mask = df["label"].isin(["up", "down"])
    y_stage2_all = np.where(df["label"] == "up", 1, 0)
    # Only compute stage2 labels for DE proteins (avoids labeling unchanged as "down")
    y_stage2 = y_stage2_all

    protein_ids = df["UniProt_ID"].values

    print(f"Loaded {X.shape[0]} proteins, {X.shape[1]} features")
    print(f"  Stage 1: {np.sum(y_stage1==0)} unchanged, {np.sum(y_stage1==1)} DE")
    print(f"  Stage 2: {np.sum(de_mask & (df['label']=='up'))} up, "
          f"{np.sum(de_mask & (df['label']=='down'))} down")

    return X, y_stage1, y_stage2, de_mask.values, feature_cols, protein_ids


def get_protein_groups(X, feature_cols, n_min_groups=20):
    """Cluster proteins into families for grouped CV.

    Uses only sequence-derived features (AAC_* and DC_*) with cosine
    distance to approximate protein family membership (~50 % seq identity).
    """
    # Extract sequence features
    seq_idx = [i for i, c in enumerate(feature_cols)
               if c.startswith("AAC_") or c.startswith("DC_")]
    if len(seq_idx) == 0:
        print("  WARNING: No AAC_/DC_ columns found; falling back to all features")
        X_seq = X
    else:
        X_seq = X[:, seq_idx]
        print(f"  Using {len(seq_idx)} sequence features (AAC + DC) for clustering")

    n = X_seq.shape[0]
    if n > 3000:
        # For large datasets, use faster approximate grouping
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = max(n_min_groups, n // 5)
        groups = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_seq)
    else:
        cos_dist = pdist(X_seq, metric="cosine")
        cos_dist = np.nan_to_num(cos_dist, nan=1.0)
        dist_matrix = squareform(cos_dist)
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=0.5,
            metric="precomputed", linkage="average"
        )
        groups = clustering.fit_predict(dist_matrix)

    print(f"  Protein groups for CV: {len(np.unique(groups))}")
    return groups


def get_cv_splitter(groups, y, n_folds, seed):
    """Get appropriate CV splitter based on group count."""
    n_unique = len(np.unique(groups))
    if n_unique >= n_folds * 2:
        cv = GroupKFold(n_splits=n_folds)
        return list(cv.split(np.zeros(len(y)), y, groups))
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y))


def optimize_xgb(X, y, groups, n_folds, n_trials, seed, stage_name):
    """Optimize XGBoost hyperparameters with Optuna."""
    print(f"\nOptimizing {stage_name} ({n_trials} trials)...")

    splits = get_cv_splitter(groups, y, n_folds, seed)
    scale_pos = np.sum(y == 0) / max(np.sum(y == 1), 1)

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": scale_pos,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": seed,
            "n_jobs": -1,
            "verbosity": 0,
        }

        aucs = []
        for train_idx, val_idx in splits:
            params_copy = {**params, "early_stopping_rounds": 50}
            clf = xgb.XGBClassifier(**params_copy)
            clf.fit(X[train_idx], y[train_idx],
                    eval_set=[(X[val_idx], y[val_idx])],
                    verbose=False)
            y_proba = clf.predict_proba(X[val_idx])[:, 1]
            try:
                auc = roc_auc_score(y[val_idx], y_proba)
            except ValueError:
                auc = 0.5
            aucs.append(auc)

        return np.mean(aucs)

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    best["scale_pos_weight"] = scale_pos
    best["objective"] = "binary:logistic"
    best["eval_metric"] = "auc"
    best["random_state"] = seed
    best["n_jobs"] = -1
    best["verbosity"] = 0

    print(f"  Best AUC: {study.best_value:.4f}")
    print(f"  Best params: max_depth={best['max_depth']}, "
          f"lr={best['learning_rate']:.4f}, n_est={best['n_estimators']}")

    return best, study.best_value


def train_final_model(X, y, params, groups, n_folds, seed, stage_name):
    """Train final model and compute CV metrics."""
    print(f"\nTraining final {stage_name} model...")

    splits = get_cv_splitter(groups, y, n_folds, seed)

    # CV evaluation
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_fold_ids = []   # track which fold each sample belongs to
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        clf = xgb.XGBClassifier(**params)
        clf.fit(X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False)

        y_pred = clf.predict(X[val_idx])
        y_proba = clf.predict_proba(X[val_idx])[:, 1]
        y_true = y[val_idx]

        try:
            auc = roc_auc_score(y_true, y_proba)
        except ValueError:
            auc = 0.5

        fold_aucs.append(auc)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_fold_ids.extend([fold] * len(y_true))

        print(f"  Fold {fold+1}: AUC={auc:.3f}")

    # Train on all data for final model
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)

    cv_metrics = {
        "mean_auc": np.mean(fold_aucs),
        "std_auc": np.std(fold_aucs),
        "fold_aucs": fold_aucs,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "y_proba": np.array(all_y_proba),
        "fold_ids": np.array(all_fold_ids),
    }

    overall_auc = roc_auc_score(cv_metrics["y_true"], cv_metrics["y_proba"])
    overall_f1 = f1_score(cv_metrics["y_true"], cv_metrics["y_pred"])
    overall_mcc = matthews_corrcoef(cv_metrics["y_true"], cv_metrics["y_pred"])

    cv_metrics["overall_auc"] = overall_auc
    cv_metrics["overall_f1"] = overall_f1
    cv_metrics["overall_mcc"] = overall_mcc

    print(f"  CV Mean AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
    print(f"  Overall AUC: {overall_auc:.3f}, F1: {overall_f1:.3f}, MCC: {overall_mcc:.3f}")

    return final_model, cv_metrics


def plot_training_results(s1_metrics, s2_metrics, s1_model, s2_model,
                          feature_cols, output_dir):
    """Generate training result figures."""
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metrics, title, labels in [
        (axes[0], s1_metrics, "Stage 1: DE vs Unchanged", ["Unchanged", "DE"]),
        (axes[1], s2_metrics, "Stage 2: Up vs Down", ["Down", "Up"]),
    ]:
        if metrics is not None:
            cm = confusion_matrix(metrics["y_true"], metrics["y_pred"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"{title}\nAUC={metrics['overall_auc']:.3f}")
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)

    plt.tight_layout()
    plt.savefig(fig_dir / "training_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Feature importance (top 30 from Stage 1)
    importances = s1_model.feature_importances_
    top_k = min(30, len(importances))
    top_idx = np.argsort(importances)[-top_k:]

    fig, ax = plt.subplots(figsize=(8, max(6, top_k * 0.3)))
    ax.barh(range(top_k), importances[top_idx], color="steelblue")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([feature_cols[i] for i in top_idx], fontsize=8)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("Stage 1: Top Features (DE vs Unchanged)")
    plt.tight_layout()
    plt.savefig(fig_dir / "training_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Figures saved to {fig_dir}/")


def generate_training_report(s1_params, s1_metrics, s2_params, s2_metrics,
                              feature_cols, s1_model, output_path):
    """Generate training report."""
    lines = []
    lines.append("# ProtDynPredict — Training Report\n")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Stage 1
    lines.append("## Stage 1: DE vs Unchanged\n")
    lines.append(f"- **CV AUC**: {s1_metrics['mean_auc']:.3f} ± {s1_metrics['std_auc']:.3f}")
    lines.append(f"- **Overall AUC**: {s1_metrics['overall_auc']:.3f}")
    lines.append(f"- **F1**: {s1_metrics['overall_f1']:.3f}")
    lines.append(f"- **MCC**: {s1_metrics['overall_mcc']:.3f}\n")

    lines.append("### Hyperparameters\n")
    for k, v in s1_params.items():
        if k not in ("objective", "eval_metric", "n_jobs", "verbosity", "random_state"):
            lines.append(f"- `{k}`: {v}")
    lines.append("")

    # Stage 2
    if s2_metrics is not None:
        lines.append("## Stage 2: Up vs Down\n")
        lines.append(f"- **CV AUC**: {s2_metrics['mean_auc']:.3f} ± {s2_metrics['std_auc']:.3f}")
        lines.append(f"- **Overall AUC**: {s2_metrics['overall_auc']:.3f}")
        lines.append(f"- **F1**: {s2_metrics['overall_f1']:.3f}")
        lines.append(f"- **MCC**: {s2_metrics['overall_mcc']:.3f}\n")

    # Top features
    lines.append("## Top 20 Features (Stage 1)\n")
    importances = s1_model.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]
    lines.append("| Rank | Feature | Importance |")
    lines.append("|------|---------|------------|")
    for rank, idx in enumerate(top_idx, 1):
        lines.append(f"| {rank} | `{feature_cols[idx]}` | {importances[idx]:.4f} |")

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\n  Report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train two-stage XGBoost model")
    parser.add_argument("--dataset", default="ucec", help="Dataset name (e.g., ucec, coad, brca)")
    args = parser.parse_args()
    dataset = args.dataset

    CONFIG["input_file"] = f"data/{dataset}/processed/feature_matrix_train.csv"
    CONFIG["model_dir"] = f"models/{dataset}"
    CONFIG["results_dir"] = f"results/{dataset}"
    CONFIG["figures_dir"] = f"results/{dataset}/figures"
    CONFIG["reports_dir"] = f"results/{dataset}/reports"

    print("=" * 60)
    print(f"  Phase 3: MODEL TRAINING (Two-Stage XGBoost) [{dataset.upper()}]")
    print("=" * 60)

    # Load data
    X, y_s1, y_s2, de_mask, feature_cols, protein_ids = load_and_prepare(CONFIG["input_file"])

    # Protein family groups
    groups = get_protein_groups(X, feature_cols)
    groups_de = groups[de_mask]

    # --- Stage 1: DE vs Unchanged ---
    s1_params, s1_best_auc = optimize_xgb(
        X, y_s1, groups, CONFIG["n_folds"],
        CONFIG["optuna_trials"], CONFIG["random_state"], "Stage 1 (DE vs Unchanged)"
    )

    s1_model, s1_metrics = train_final_model(
        X, y_s1, s1_params, groups, CONFIG["n_folds"],
        CONFIG["random_state"], "Stage 1"
    )

    # --- Stage 2: Up vs Down (DE proteins only) ---
    X_de = X[de_mask]
    y_de = y_s2[de_mask]

    s2_params = None
    s2_metrics = None
    s2_model = None

    if len(X_de) >= 30:
        s2_params, s2_best_auc = optimize_xgb(
            X_de, y_de, groups_de, CONFIG["n_folds"],
            CONFIG["optuna_trials"], CONFIG["random_state"], "Stage 2 (Up vs Down)"
        )

        s2_model, s2_metrics = train_final_model(
            X_de, y_de, s2_params, groups_de, CONFIG["n_folds"],
            CONFIG["random_state"], "Stage 2"
        )
    else:
        print(f"\n  Skipping Stage 2: only {len(X_de)} DE proteins (need >= 30)")

    # --- Save models ---
    model_dir = Path(CONFIG["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": s1_model,
        "params": s1_params,
        "feature_cols": feature_cols,
        "metrics": {k: v for k, v in s1_metrics.items()
                    if k not in ("y_true", "y_pred", "y_proba", "fold_ids")},
    }, model_dir / "stage1_model.joblib")

    if s2_model is not None:
        joblib.dump({
            "model": s2_model,
            "params": s2_params,
            "feature_cols": feature_cols,
            "metrics": {k: v for k, v in s2_metrics.items()
                        if k not in ("y_true", "y_pred", "y_proba", "fold_ids")},
        }, model_dir / "stage2_model.joblib")

    print(f"\n  Models saved to {model_dir}/")

    # --- Save CV predictions ---
    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        results_dir / "cv_predictions_stage1.npz",
        y_true=s1_metrics["y_true"],
        y_proba=s1_metrics["y_proba"],
        y_pred=s1_metrics["y_pred"],
        fold_aucs=np.array(s1_metrics["fold_aucs"]),
        fold_ids=s1_metrics["fold_ids"],
    )
    print(f"  Stage 1 CV predictions saved to {results_dir / 'cv_predictions_stage1.npz'}")

    if s2_metrics is not None:
        np.savez(
            results_dir / "cv_predictions_stage2.npz",
            y_true=s2_metrics["y_true"],
            y_proba=s2_metrics["y_proba"],
            y_pred=s2_metrics["y_pred"],
            fold_aucs=np.array(s2_metrics["fold_aucs"]),
            fold_ids=s2_metrics["fold_ids"],
        )
        print(f"  Stage 2 CV predictions saved to {results_dir / 'cv_predictions_stage2.npz'}")

    # --- Plots and report ---
    plot_training_results(s1_metrics, s2_metrics, s1_model, s2_model,
                          feature_cols, CONFIG["figures_dir"])

    generate_training_report(
        s1_params, s1_metrics, s2_params, s2_metrics,
        feature_cols, s1_model,
        str(Path(CONFIG["reports_dir"]) / "training_report.md")
    )

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Stage 1 AUC: {s1_metrics['overall_auc']:.3f}")
    if s2_metrics:
        print(f"  Stage 2 AUC: {s2_metrics['overall_auc']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
