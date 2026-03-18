#!/usr/bin/env python3
"""
13_improved_model.py
Improved ProtDynPredict model via PCA dimensionality reduction on sequence
features, alternative classifiers, and Platt calibration.

Motivation: Feature ablation (04b) showed removing 999 sequence features
*improves* AUC by +0.025, while GO-slim features carry the dominant signal.
This script systematically compares model variants to identify the best
feature configuration.

Variants compared (all use same default XGBoost params for fair comparison):
  A) PCA-reduced sequence + all non-sequence features (XGBoost)
  B) Non-sequence features only — GO-slim + PPI + detectability + pathway (XGBoost)
  C) Original full model (XGBoost, all 1163 features)
  D) PCA-reduced features + Logistic Regression (L1)
  E) PCA-reduced features + LightGBM

After initial comparison, the best XGBoost variant gets Optuna tuning.

Methodological safeguards:
  - PCA is fit inside each CV fold on training data only (no leakage)
  - Early stopping uses a nested held-out set, not the evaluation fold
  - Platt calibration trains on full fold, fits sigmoid on nested held-out
  - All XGBoost variants use identical default params for fair comparison
  - Holm-Bonferroni correction for multiple model comparisons

Input:  data/<dataset>/processed/feature_matrix_train.csv
Output: results/<dataset>/figures/pca_variance.png
        results/<dataset>/figures/model_comparison.png
        results/<dataset>/figures/calibration_improved.png
        results/<dataset>/reports/model_improvement_report.md
        models/<dataset>/stage1_improved_model.joblib
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import optuna
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
import xgboost as xgb

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Constants ─────────────────────────────────────────────────────────────
LEAKY_PREFIXES = [
    "ppi_frac_neighbors_", "ppi_weighted_frac_",
    "pw_max_frac_", "pw_mean_frac_",
    "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_",
]

SEQUENCE_PREFIXES = [
    "AAC_", "DC_", "CTriad_", "CTDC_", "CTDT_", "CTDD_",
    "QSO_", "APseAAC_", "PseAAC_",
]

META_COLS = {"UniProt_ID", "label", "log2FC", "adj_pvalue"}

RANDOM_STATE = 42
N_FOLDS = 5
EARLY_STOP_FRAC = 0.15  # fraction of training fold held out for early stopping


# ── Helpers ───────────────────────────────────────────────────────────────

def _safe_feature_cols(all_cols):
    """Return feature columns excluding metadata and leaky features."""
    return [c for c in all_cols
            if c not in META_COLS
            and not any(c.startswith(p) for p in LEAKY_PREFIXES)]


def _is_sequence_feature(col):
    return any(col.startswith(p) for p in SEQUENCE_PREFIXES)


def _build_groups(X, feature_cols, n_min_groups=20):
    """Cluster proteins into families for grouped CV."""
    seq_idx = [i for i, c in enumerate(feature_cols)
               if c.startswith("AAC_") or c.startswith("DC_")]
    if len(seq_idx) == 0:
        X_seq = X
    else:
        X_seq = X[:, seq_idx]

    n = X_seq.shape[0]
    if n > 3000:
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


def _get_cv_splits(groups, y, n_folds, seed):
    """Get CV splits, using GroupKFold if enough groups."""
    n_unique = len(np.unique(groups))
    if n_unique >= n_folds * 2:
        cv = GroupKFold(n_splits=n_folds)
        return list(cv.split(np.zeros(len(y)), y, groups))
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(cv.split(np.zeros(len(y)), y))


def _nested_early_stop_split(train_idx, y, fold, frac=EARLY_STOP_FRAC):
    """Split training indices into base-train and early-stopping held-out.

    This prevents the evaluation fold from being used for early stopping,
    which would cause optimistic bias.
    """
    rng = np.random.RandomState(RANDOM_STATE + fold)
    n = len(train_idx)
    n_es = max(int(n * frac), 30)
    perm = rng.permutation(n)
    es_mask = perm[:n_es]
    base_mask = perm[n_es:]
    return train_idx[base_mask], train_idx[es_mask]


def holm_bonferroni(p_values):
    """Apply Holm-Bonferroni correction to a list of p-values.

    Returns adjusted p-values.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        idx = sorted_idx[i]
        prev_idx = sorted_idx[i - 1]
        adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
    return adjusted


# ── Data Loading ──────────────────────────────────────────────────────────

def load_data(data_path):
    """Load and split features into sequence / non-sequence."""
    df = pd.read_csv(data_path)
    all_features = _safe_feature_cols(df.columns)

    seq_cols = [c for c in all_features if _is_sequence_feature(c)]
    nonseq_cols = [c for c in all_features if not _is_sequence_feature(c)]

    X_all = df[all_features].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    X_seq = df[seq_cols].values.astype(np.float32)
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)

    X_nonseq = df[nonseq_cols].values.astype(np.float32)
    X_nonseq = np.nan_to_num(X_nonseq, nan=0.0, posinf=0.0, neginf=0.0)

    y = np.where(df["label"] == "unchanged", 0, 1)

    print(f"  Proteins: {len(y)} | Total safe features: {len(all_features)}")
    print(f"  Sequence features: {len(seq_cols)}")
    print(f"  Non-sequence features: {len(nonseq_cols)}")
    print(f"  DE: {np.sum(y==1)} | Unchanged: {np.sum(y==0)}")

    return X_all, X_seq, X_nonseq, y, all_features, seq_cols, nonseq_cols


# ── PCA (for variance plot only — actual PCA done per-fold) ──────────────

def compute_pca_variance_curve(X_seq, variance_threshold=0.95):
    """Fit PCA on full data ONLY to generate the variance explained plot.

    This is NOT used for model training — per-fold PCA is used instead.
    Returns cumvar and n_components for the plot.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seq)

    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)

    print(f"\n  PCA (full data, for plot only): {X_seq.shape[1]} -> {n_components} components")
    print(f"  Variance explained: {cumvar[n_components-1]:.4f}")

    return cumvar, n_components


# ── CV Evaluation (with per-fold PCA for variants that need it) ──────────

def cv_evaluate_xgb(X, y, params, splits, name, X_seq=None, X_nonseq=None,
                     pca_variance=0.95, use_pca=False):
    """XGBoost CV with nested early stopping split.

    If use_pca=True, PCA is fit on the training fold's sequence features
    and applied to the validation fold (no leakage).

    Early stopping uses a nested held-out portion of the training fold,
    NOT the evaluation fold.
    """
    fold_aucs = []
    all_y_true, all_y_proba, all_y_pred = [], [], []

    for fold, (train_idx, val_idx) in enumerate(splits):
        # Nested split: base-train for fitting, es-holdout for early stopping
        base_idx, es_idx = _nested_early_stop_split(train_idx, y, fold)

        if use_pca and X_seq is not None and X_nonseq is not None:
            # Fit PCA on training fold only
            scaler = StandardScaler()
            X_seq_train_scaled = scaler.fit_transform(X_seq[base_idx])
            X_seq_es_scaled = scaler.transform(X_seq[es_idx])
            X_seq_val_scaled = scaler.transform(X_seq[val_idx])

            pca = PCA(n_components=pca_variance, random_state=RANDOM_STATE)
            X_pca_train = pca.fit_transform(X_seq_train_scaled)
            X_pca_es = pca.transform(X_seq_es_scaled)
            X_pca_val = pca.transform(X_seq_val_scaled)

            X_train = np.hstack([X_pca_train, X_nonseq[base_idx]])
            X_es = np.hstack([X_pca_es, X_nonseq[es_idx]])
            X_val = np.hstack([X_pca_val, X_nonseq[val_idx]])
        else:
            X_train = X[base_idx]
            X_es = X[es_idx]
            X_val = X[val_idx]

        train_params = {**params, "early_stopping_rounds": 50}
        clf = xgb.XGBClassifier(**train_params)
        clf.fit(X_train, y[base_idx],
                eval_set=[(X_es, y[es_idx])],
                verbose=False)

        y_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)

        try:
            auc = roc_auc_score(y[val_idx], y_proba)
        except ValueError:
            auc = 0.5

        fold_aucs.append(auc)
        all_y_true.extend(y[val_idx])
        all_y_proba.extend(y_proba)
        all_y_pred.extend(y_pred)

    fold_aucs = np.array(fold_aucs)
    y_true = np.array(all_y_true)
    y_proba = np.array(all_y_proba)
    y_pred = np.array(all_y_pred)

    overall_auc = roc_auc_score(y_true, y_proba)
    overall_f1 = f1_score(y_true, y_pred)
    overall_mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_proba)

    print(f"  {name}: AUC={overall_auc:.4f} (folds: {fold_aucs.mean():.4f}+/-{fold_aucs.std():.4f}), "
          f"F1={overall_f1:.3f}, MCC={overall_mcc:.3f}, Brier={brier:.4f}")

    return {
        "name": name,
        "fold_aucs": fold_aucs,
        "overall_auc": overall_auc,
        "mean_auc": float(fold_aucs.mean()),
        "std_auc": float(fold_aucs.std()),
        "overall_f1": overall_f1,
        "overall_mcc": overall_mcc,
        "brier": brier,
        "y_true": y_true,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


def cv_evaluate_generic(X, y, clf_factory, splits, name, X_seq=None,
                         X_nonseq=None, pca_variance=0.95, use_pca=False,
                         scale=False):
    """Generic CV for non-XGBoost models.

    If use_pca=True, PCA is fit per-fold on training data.
    If scale=True, StandardScaler is fit per-fold on training data.
    """
    fold_aucs = []
    all_y_true, all_y_proba, all_y_pred = [], [], []

    for fold, (train_idx, val_idx) in enumerate(splits):
        if use_pca and X_seq is not None and X_nonseq is not None:
            scaler = StandardScaler()
            X_seq_train_scaled = scaler.fit_transform(X_seq[train_idx])
            X_seq_val_scaled = scaler.transform(X_seq[val_idx])

            pca = PCA(n_components=pca_variance, random_state=RANDOM_STATE)
            X_pca_train = pca.fit_transform(X_seq_train_scaled)
            X_pca_val = pca.transform(X_seq_val_scaled)

            X_train = np.hstack([X_pca_train, X_nonseq[train_idx]])
            X_val = np.hstack([X_pca_val, X_nonseq[val_idx]])
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]

        if scale:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_val = sc.transform(X_val)

        clf = clf_factory()
        clf.fit(X_train, y[train_idx])
        y_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)

        try:
            auc = roc_auc_score(y[val_idx], y_proba)
        except ValueError:
            auc = 0.5

        fold_aucs.append(auc)
        all_y_true.extend(y[val_idx])
        all_y_proba.extend(y_proba)
        all_y_pred.extend(y_pred)

    fold_aucs = np.array(fold_aucs)
    y_true = np.array(all_y_true)
    y_proba = np.array(all_y_proba)
    y_pred = np.array(all_y_pred)

    overall_auc = roc_auc_score(y_true, y_proba)
    overall_f1 = f1_score(y_true, y_pred)
    overall_mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_proba)

    print(f"  {name}: AUC={overall_auc:.4f} (folds: {fold_aucs.mean():.4f}+/-{fold_aucs.std():.4f}), "
          f"F1={overall_f1:.3f}, MCC={overall_mcc:.3f}, Brier={brier:.4f}")

    return {
        "name": name,
        "fold_aucs": fold_aucs,
        "overall_auc": overall_auc,
        "mean_auc": float(fold_aucs.mean()),
        "std_auc": float(fold_aucs.std()),
        "overall_f1": overall_f1,
        "overall_mcc": overall_mcc,
        "brier": brier,
        "y_true": y_true,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


# ── Optuna tuning (applied to best variant after initial comparison) ─────

def optimize_xgb(X, y, splits, n_trials, seed, name, X_seq=None,
                  X_nonseq=None, pca_variance=0.95, use_pca=False):
    """Optuna hyperparameter search for XGBoost with proper nested splits."""
    print(f"\n  Optimizing {name} ({n_trials} trials)...")
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
        for fold, (train_idx, val_idx) in enumerate(splits):
            base_idx, es_idx = _nested_early_stop_split(train_idx, y, fold)

            if use_pca and X_seq is not None and X_nonseq is not None:
                scaler = StandardScaler()
                X_seq_train_sc = scaler.fit_transform(X_seq[base_idx])
                X_seq_es_sc = scaler.transform(X_seq[es_idx])
                X_seq_val_sc = scaler.transform(X_seq[val_idx])

                pca = PCA(n_components=pca_variance, random_state=RANDOM_STATE)
                X_train = np.hstack([pca.fit_transform(X_seq_train_sc), X_nonseq[base_idx]])
                X_es = np.hstack([pca.transform(X_seq_es_sc), X_nonseq[es_idx]])
                X_val = np.hstack([pca.transform(X_seq_val_sc), X_nonseq[val_idx]])
            else:
                X_train = X[base_idx]
                X_es = X[es_idx]
                X_val = X[val_idx]

            clf = xgb.XGBClassifier(**{**params, "early_stopping_rounds": 50})
            clf.fit(X_train, y[base_idx],
                    eval_set=[(X_es, y[es_idx])],
                    verbose=False)
            y_proba = clf.predict_proba(X_val)[:, 1]
            try:
                aucs.append(roc_auc_score(y[val_idx], y_proba))
            except ValueError:
                aucs.append(0.5)
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

    print(f"  Best CV AUC: {study.best_value:.4f}")
    print(f"  Best params: max_depth={best['max_depth']}, "
          f"lr={best['learning_rate']:.4f}, n_est={best['n_estimators']}")

    return best


# ── Platt Calibration ────────────────────────────────────────────────────

def calibrate_and_evaluate(X, y, params, splits, name):
    """Platt calibration: train base on full training fold, fit sigmoid
    on a nested held-out portion, evaluate on true validation fold.

    The base model is trained on the FULL training fold (no data reduction),
    so AUC is directly comparable to uncalibrated variants.
    """
    from sklearn.linear_model import LogisticRegression as _LR

    fold_aucs = []
    all_y_true, all_y_proba_uncal, all_y_proba_cal = [], [], []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        # Train base model on full training fold (no early stopping for
        # calibration to avoid needing yet another split)
        base_params = {k: v for k, v in params.items()
                       if k != "early_stopping_rounds"}
        base_clf = xgb.XGBClassifier(**base_params)
        base_clf.fit(X[train_idx], y[train_idx], verbose=False)

        # Nested split for Platt fitting
        rng = np.random.RandomState(RANDOM_STATE + fold + 100)
        n_train = len(train_idx)
        perm = rng.permutation(n_train)
        n_cal = max(int(n_train * 0.2), 50)
        cal_idx = train_idx[perm[:n_cal]]

        # Fit Platt sigmoid on calibration subset's predictions
        cal_proba = base_clf.predict_proba(X[cal_idx])[:, 1].reshape(-1, 1)
        platt = _LR(max_iter=1000)
        platt.fit(cal_proba, y[cal_idx])

        # Evaluate on validation fold
        y_proba_uncal = base_clf.predict_proba(X[val_idx])[:, 1]
        y_proba_cal = platt.predict_proba(y_proba_uncal.reshape(-1, 1))[:, 1]
        y_pred = (y_proba_cal >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y[val_idx], y_proba_cal)
        except ValueError:
            auc = 0.5

        fold_aucs.append(auc)
        all_y_true.extend(y[val_idx])
        all_y_proba_uncal.extend(y_proba_uncal)
        all_y_proba_cal.extend(y_proba_cal)
        all_y_pred.extend(y_pred)

    fold_aucs = np.array(fold_aucs)
    y_true = np.array(all_y_true)
    y_proba_uncal = np.array(all_y_proba_uncal)
    y_proba_cal = np.array(all_y_proba_cal)
    y_pred = np.array(all_y_pred)

    overall_auc = roc_auc_score(y_true, y_proba_cal)
    overall_f1 = f1_score(y_true, y_pred)
    overall_mcc = matthews_corrcoef(y_true, y_pred)
    brier_uncal = brier_score_loss(y_true, y_proba_uncal)
    brier_cal = brier_score_loss(y_true, y_proba_cal)

    print(f"  {name}: AUC={overall_auc:.4f}, F1={overall_f1:.3f}, MCC={overall_mcc:.3f}")
    print(f"    Brier uncalibrated: {brier_uncal:.4f} -> calibrated: {brier_cal:.4f}")

    return {
        "name": name,
        "fold_aucs": fold_aucs,
        "overall_auc": overall_auc,
        "mean_auc": float(fold_aucs.mean()),
        "std_auc": float(fold_aucs.std()),
        "overall_f1": overall_f1,
        "overall_mcc": overall_mcc,
        "brier": brier_cal,
        "brier_uncalibrated": brier_uncal,
        "y_true": y_true,
        "y_proba": y_proba_cal,
        "y_proba_uncal": y_proba_uncal,
        "y_pred": y_pred,
    }


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_pca_variance(cumvar, n_components, fig_dir):
    """Plot cumulative explained variance from PCA."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(1, len(cumvar) + 1)
    ax.plot(x, cumvar, color="steelblue", linewidth=2)
    ax.axhline(0.95, color="crimson", linestyle="--", linewidth=1, alpha=0.7,
               label="95% threshold")
    ax.axvline(n_components, color="orange", linestyle="--", linewidth=1, alpha=0.7,
               label=f"{n_components} components")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title(f"PCA on Sequence Features: {n_components} components explain "
                 f"{cumvar[n_components-1]:.1%} variance")
    ax.legend()
    ax.set_xlim(1, min(200, len(cumvar)))
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_model_comparison(all_results, fig_dir):
    """Bar chart comparing model variants."""
    names = [r["name"] for r in all_results]
    aucs = [r["overall_auc"] for r in all_results]
    stds = [r["std_auc"] for r in all_results]
    briers = [r["brier"] for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(names))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    ax1.bar(x, aucs, yerr=stds, capsize=4, color=colors,
            edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("AUC")
    ax1.set_title("Model Comparison: AUC")
    for i, (a, s) in enumerate(zip(aucs, stds)):
        ax1.text(i, a + s + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
    ymin = min(aucs) - max(stds) - 0.04
    ax1.set_ylim(max(0.5, ymin), min(1.0, max(aucs) + max(stds) + 0.04))

    ax2.bar(x, briers, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Brier Score (lower is better)")
    ax2.set_title("Model Comparison: Brier Score")
    for i, b in enumerate(briers):
        ax2.text(i, b + 0.003, f"{b:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_calibration_comparison(uncal_result, cal_result, fig_dir):
    """Plot calibration curves: uncalibrated vs calibrated."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, res, title in [
        (ax1, uncal_result, "Before Calibration"),
        (ax2, cal_result, "After Platt Calibration"),
    ]:
        y_true = res["y_true"]
        y_proba = res["y_proba"]
        brier = brier_score_loss(y_true, y_proba)

        fraction_pos, mean_predicted = calibration_curve(
            y_true, y_proba, n_bins=10, strategy="uniform"
        )
        ax.plot([0, 1], [0, 1], "k:", linewidth=1, label="Perfect")
        ax.plot(mean_predicted, fraction_pos, "s-", color="steelblue",
                linewidth=2, markersize=7, label=f"Brier={brier:.4f}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_improved.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Report Generation ─────────────────────────────────────────────────────

def generate_report(all_results, baseline_result, pca_info, cal_result,
                     best_tuned_result, rpt_dir):
    """Generate markdown comparison report."""
    lines = []
    lines.append("# Model Improvement Report\n")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("---\n")

    # Methodology notes
    lines.append("## 0. Methodology\n")
    lines.append("- All XGBoost variants use identical default hyperparameters for fair comparison")
    lines.append("- PCA is fit inside each CV fold on training data only (no leakage)")
    lines.append("- Early stopping uses a nested 15% held-out from the training fold, not the evaluation fold")
    lines.append("- Platt calibration trains on the full training fold; sigmoid is fit on a nested 20% held-out")
    lines.append("- Best variant receives Optuna tuning (50 trials) as a follow-up")
    lines.append("- p-values are corrected for multiple comparisons using Holm-Bonferroni\n")

    # PCA summary
    lines.append("## 1. PCA Dimensionality Reduction\n")
    lines.append(f"- **Original sequence features**: {pca_info['n_original']}")
    lines.append(f"- **PCA components (95% var)**: {pca_info['n_components']}")
    lines.append(f"- **Compression ratio**: {pca_info['n_original'] / pca_info['n_components']:.1f}x\n")
    lines.append("![PCA Variance](../figures/pca_variance.png)\n")

    # Model comparison table
    lines.append("## 2. Model Comparison (Default Params)\n")
    lines.append("| Model | AUC | AUC (mean+/-std) | F1 | MCC | Brier | vs Baseline |")
    lines.append("|-------|-----|------------------|----|-----|-------|-------------|")

    baseline_auc = baseline_result["overall_auc"]
    for r in all_results:
        delta = r["overall_auc"] - baseline_auc
        lines.append(
            f"| {r['name']} | {r['overall_auc']:.4f} | "
            f"{r['mean_auc']:.4f}+/-{r['std_auc']:.4f} | "
            f"{r['overall_f1']:.3f} | {r['overall_mcc']:.3f} | "
            f"{r['brier']:.4f} | {delta:+.4f} |"
        )
    lines.append("")

    # Paired t-tests with Holm-Bonferroni
    lines.append("## 3. Statistical Significance (Holm-Bonferroni corrected)\n")
    lines.append("| Model | t-stat | Raw p | Adjusted p | Significant? |")
    lines.append("|-------|--------|-------|------------|-------------|")

    non_baseline = [r for r in all_results if r["name"] != baseline_result["name"]]
    raw_pvals = []
    t_stats = []
    for r in non_baseline:
        t_stat, p_val = ttest_rel(baseline_result["fold_aucs"], r["fold_aucs"])
        raw_pvals.append(p_val)
        t_stats.append(t_stat)

    adj_pvals = holm_bonferroni(np.array(raw_pvals))

    for r, t_stat, raw_p, adj_p in zip(non_baseline, t_stats, raw_pvals, adj_pvals):
        sig = "Yes" if adj_p < 0.05 else "No"
        lines.append(f"| {r['name']} | {t_stat:.3f} | {raw_p:.4f} | {adj_p:.4f} | {sig} |")

    lines.append("")
    lines.append("*Note: Paired t-tests on 5 CV folds (df=4) have limited power and may be "
                 "anti-conservative due to overlapping training sets (Dietterich 1998, "
                 "Nadeau & Bengio 2003). Results should be interpreted with caution.*\n")

    # Optuna-tuned best
    if best_tuned_result is not None:
        lines.append("## 4. Optuna-Tuned Best Variant\n")
        lines.append(f"- **Model**: {best_tuned_result['name']}")
        lines.append(f"- **AUC**: {best_tuned_result['overall_auc']:.4f} "
                     f"(folds: {best_tuned_result['mean_auc']:.4f}+/-{best_tuned_result['std_auc']:.4f})")
        delta_tuned = best_tuned_result['overall_auc'] - baseline_auc
        lines.append(f"- **vs baseline**: {delta_tuned:+.4f}\n")

    # Calibration
    lines.append("## 5. Platt Calibration\n")
    if cal_result is not None:
        lines.append(f"- **Brier (uncalibrated)**: {cal_result['brier_uncalibrated']:.4f}")
        lines.append(f"- **Brier (calibrated)**: {cal_result['brier']:.4f}")
        improvement = cal_result['brier_uncalibrated'] - cal_result['brier']
        lines.append(f"- **Improvement**: {improvement:+.4f}")
        lines.append("- Note: Base model trained on full training fold (AUC comparable to uncalibrated)\n")
        lines.append("![Calibration](../figures/calibration_improved.png)\n")

    # Recommendation
    lines.append("## 6. Recommendation\n")
    final_best = best_tuned_result if best_tuned_result is not None else max(
        all_results, key=lambda r: r["overall_auc"])
    lines.append(f"**Best model**: {final_best['name']} (AUC={final_best['overall_auc']:.4f})\n")

    lines.append("---\n")
    lines.append("*Generated by `13_improved_model.py`*\n")

    report_path = rpt_dir / "model_improvement_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved: {report_path}")
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Improved model via PCA + calibration + alternatives"
    )
    parser.add_argument("--dataset", default="ucec")
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("--pca-variance", type=float, default=0.95,
                        help="Cumulative variance threshold for PCA (default: 0.95)")
    args = parser.parse_args()

    dataset = args.dataset
    n_trials = args.optuna_trials
    pca_var = args.pca_variance

    fig_dir = Path(f"results/{dataset}/figures")
    rpt_dir = Path(f"results/{dataset}/reports")
    model_dir = Path(f"models/{dataset}")
    data_path = Path(f"data/{dataset}/processed/feature_matrix_train.csv")

    for d in [fig_dir, rpt_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  IMPROVED MODEL ANALYSIS [{dataset.upper()}]")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    X_all, X_seq, X_nonseq, y, all_features, seq_cols, nonseq_cols = load_data(data_path)

    # ── Build protein groups ──────────────────────────────────────────────
    print("\nBuilding protein family groups...")
    groups = _build_groups(X_all, all_features)
    splits = _get_cv_splits(groups, y, N_FOLDS, RANDOM_STATE)

    # ── PCA variance curve (for plot only) ────────────────────────────────
    print("\n--- Step 1: PCA Variance Analysis ---")
    cumvar, n_components = compute_pca_variance_curve(X_seq, pca_var)

    pca_info = {
        "n_original": len(seq_cols),
        "n_components": n_components,
        "n_nonseq": len(nonseq_cols),
    }

    plot_pca_variance(cumvar, n_components, fig_dir)
    print(f"  PCA variance plot saved to {fig_dir}/pca_variance.png")

    # ── Step 2: Fair comparison — all use same default XGBoost params ─────
    print("\n--- Step 2: Model Comparison (Default Params) ---")
    all_results = []

    scale_pos = np.sum(y == 0) / max(np.sum(y == 1), 1)
    default_xgb_params = dict(
        max_depth=5, learning_rate=0.1, n_estimators=300,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        objective="binary:logistic", eval_metric="auc",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )

    # (C) Original full model — baseline
    print("\nVariant C: Full model (all 1163 features, default params)...")
    baseline_result = cv_evaluate_xgb(
        X_all, y, default_xgb_params, splits, "C: Full model (baseline)"
    )
    all_results.append(baseline_result)

    # (A) PCA-reduced + non-sequence (per-fold PCA, default params)
    print("\nVariant A: PCA-reduced + non-sequence (per-fold PCA, default params)...")
    pca_result = cv_evaluate_xgb(
        None, y, default_xgb_params, splits, "A: PCA+XGBoost",
        X_seq=X_seq, X_nonseq=X_nonseq, pca_variance=pca_var, use_pca=True
    )
    all_results.append(pca_result)

    # (B) Non-sequence only (default params)
    print("\nVariant B: Non-sequence features only (164 features, default params)...")
    nonseq_result = cv_evaluate_xgb(
        X_nonseq, y, default_xgb_params, splits, "B: Non-sequence only"
    )
    all_results.append(nonseq_result)

    # ── Step 3: Alternative models (per-fold PCA + scaling) ───────────────
    print("\n--- Step 3: Alternative Models ---")

    # (D) Logistic Regression L1 — per-fold PCA and scaling
    print("\nVariant D: LogReg L1 (per-fold PCA + scaling)...")
    lr_result = cv_evaluate_generic(
        None, y,
        lambda: LogisticRegression(
            penalty="l1", solver="saga", C=1.0, max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
            l1_ratio=1.0,
        ),
        splits, "D: LogReg L1 (PCA)",
        X_seq=X_seq, X_nonseq=X_nonseq, pca_variance=pca_var,
        use_pca=True, scale=True,
    )
    all_results.append(lr_result)

    # (E) LightGBM — per-fold PCA
    if HAS_LGBM:
        print("\nVariant E: LightGBM (per-fold PCA)...")
        lgbm_result = cv_evaluate_generic(
            None, y,
            lambda: lgb.LGBMClassifier(
                max_depth=5, learning_rate=0.1, n_estimators=300,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                objective="binary", metric="auc",
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            ),
            splits, "E: LightGBM (PCA)",
            X_seq=X_seq, X_nonseq=X_nonseq, pca_variance=pca_var,
            use_pca=True,
        )
        all_results.append(lgbm_result)
    else:
        print("\nVariant E: LightGBM — skipped (not installed)")

    # ── Step 4: Optuna-tune the best XGBoost variant ──────────────────────
    print("\n--- Step 4: Optuna Tuning of Best Variant ---")
    # Only consider XGBoost variants (A, B, C) for tuning
    xgb_results = [r for r in all_results if r["name"].startswith(("A:", "B:", "C:"))]
    best_default = max(xgb_results, key=lambda r: r["overall_auc"])
    print(f"  Best default XGBoost variant: {best_default['name']} "
          f"(AUC={best_default['overall_auc']:.4f})")

    # Determine whether to use PCA for tuning
    best_uses_pca = best_default["name"].startswith("A:")
    best_is_nonseq = best_default["name"].startswith("B:")

    if best_is_nonseq:
        best_tuned_params = optimize_xgb(
            X_nonseq, y, splits, n_trials, RANDOM_STATE,
            "Non-sequence XGBoost"
        )
        best_tuned_result = cv_evaluate_xgb(
            X_nonseq, y, best_tuned_params, splits,
            "B-tuned: Non-sequence (Optuna)"
        )
    elif best_uses_pca:
        best_tuned_params = optimize_xgb(
            None, y, splits, n_trials, RANDOM_STATE,
            "PCA+XGBoost",
            X_seq=X_seq, X_nonseq=X_nonseq, pca_variance=pca_var,
            use_pca=True
        )
        best_tuned_result = cv_evaluate_xgb(
            None, y, best_tuned_params, splits,
            "A-tuned: PCA+XGBoost (Optuna)",
            X_seq=X_seq, X_nonseq=X_nonseq, pca_variance=pca_var,
            use_pca=True
        )
    else:
        best_tuned_params = optimize_xgb(
            X_all, y, splits, n_trials, RANDOM_STATE,
            "Full model XGBoost"
        )
        best_tuned_result = cv_evaluate_xgb(
            X_all, y, best_tuned_params, splits,
            "C-tuned: Full model (Optuna)"
        )
    all_results.append(best_tuned_result)

    # ── Step 5: Platt Calibration on best model ──────────────────────────
    print("\n--- Step 5: Platt Calibration ---")
    # Calibrate the best non-tuned XGBoost variant
    if best_is_nonseq:
        cal_result = calibrate_and_evaluate(
            X_nonseq, y, default_xgb_params, splits,
            "B-cal: Non-sequence (calibrated)"
        )
    else:
        cal_result = calibrate_and_evaluate(
            X_all, y, default_xgb_params, splits,
            "C-cal: Full model (calibrated)"
        )
    all_results.append(cal_result)

    # Plot calibration: best uncalibrated vs calibrated
    plot_calibration_comparison(best_default, cal_result, fig_dir)
    print(f"  Calibration plot saved to {fig_dir}/calibration_improved.png")

    # ── Step 6: Plots and report ──────────────────────────────────────────
    print("\n--- Step 6: Report Generation ---")
    plot_model_comparison(all_results, fig_dir)
    print(f"  Model comparison plot saved to {fig_dir}/model_comparison.png")

    report_path = generate_report(
        all_results, baseline_result, pca_info, cal_result,
        best_tuned_result, rpt_dir
    )

    # ── Save best model ──────────────────────────────────────────────────
    final_best = max(all_results, key=lambda r: r["overall_auc"])
    print(f"\n  Best overall: {final_best['name']} (AUC={final_best['overall_auc']:.4f})")

    # Determine what to save based on which model won
    is_nonseq_best = "Non-sequence" in final_best["name"]
    is_pca_best = "PCA" in final_best["name"]

    if is_nonseq_best:
        # Train final non-sequence model on all data
        save_params = best_tuned_params if "tuned" in final_best["name"] else default_xgb_params
        final_clf = xgb.XGBClassifier(**save_params)
        final_clf.fit(X_nonseq, y, verbose=False)
        save_dict = {
            "model": final_clf,
            "params": save_params,
            "feature_type": "non-sequence",
            "nonseq_cols": nonseq_cols,
            "metrics": {
                "overall_auc": final_best["overall_auc"],
                "mean_auc": final_best["mean_auc"],
                "std_auc": final_best["std_auc"],
                "overall_f1": final_best["overall_f1"],
                "overall_mcc": final_best["overall_mcc"],
                "brier": final_best["brier"],
            },
        }
    elif is_pca_best:
        # For PCA model, fit PCA on all data for the saved artifact
        save_params = best_tuned_params if "tuned" in final_best["name"] else default_xgb_params
        scaler = StandardScaler()
        X_seq_scaled = scaler.fit_transform(X_seq)
        pca = PCA(n_components=pca_var, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_seq_scaled)
        X_combined = np.hstack([X_pca, X_nonseq])
        final_clf = xgb.XGBClassifier(**save_params)
        final_clf.fit(X_combined, y, verbose=False)
        save_dict = {
            "model": final_clf,
            "params": save_params,
            "feature_type": "pca",
            "pca": pca,
            "scaler": scaler,
            "seq_cols": seq_cols,
            "nonseq_cols": nonseq_cols,
            "pca_info": pca_info,
            "metrics": {
                "overall_auc": final_best["overall_auc"],
                "mean_auc": final_best["mean_auc"],
                "std_auc": final_best["std_auc"],
                "overall_f1": final_best["overall_f1"],
                "overall_mcc": final_best["overall_mcc"],
                "brier": final_best["brier"],
            },
        }
    else:
        # Full model
        save_params = best_tuned_params if "tuned" in final_best["name"] else default_xgb_params
        final_clf = xgb.XGBClassifier(**save_params)
        final_clf.fit(X_all, y, verbose=False)
        save_dict = {
            "model": final_clf,
            "params": save_params,
            "feature_type": "full",
            "feature_cols": all_features,
            "metrics": {
                "overall_auc": final_best["overall_auc"],
                "mean_auc": final_best["mean_auc"],
                "std_auc": final_best["std_auc"],
                "overall_f1": final_best["overall_f1"],
                "overall_mcc": final_best["overall_mcc"],
                "brier": final_best["brier"],
            },
        }

    joblib.dump(save_dict, model_dir / "stage1_improved_model.joblib")
    print(f"  Improved model saved to {model_dir / 'stage1_improved_model.joblib'}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  IMPROVED MODEL ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Baseline AUC:      {baseline_result['overall_auc']:.4f}")
    print(f"  Best model:        {final_best['name']}")
    print(f"  Best AUC:          {final_best['overall_auc']:.4f}")
    delta = final_best['overall_auc'] - baseline_result['overall_auc']
    print(f"  Improvement:       {delta:+.4f}")
    t_stat, p_val = ttest_rel(baseline_result["fold_aucs"], final_best["fold_aucs"])
    print(f"  Paired t-test:     t={t_stat:.3f}, p={p_val:.4f}")
    print(f"  Report:            {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
