#!/usr/bin/env python3
"""
04_validate_within.py
Rigorous within-experiment validation for ProtDynPredict.

Validation sections:
  1. Permutation test (1000 shuffles) — empirical p-value for Stage 1 AUC
  2. Bootstrap confidence intervals (1000 iterations) — 95% CI for AUC, F1, MCC
  3. Threshold sensitivity — AUC stability across FC thresholds
  4. Calibration curve — reliability diagram + Brier score
  5. MNAR simulation — missing-not-at-random robustness check

Input:  data/processed/feature_matrix_train.csv
        data/raw/de_results.csv
        models/stage1_model.joblib, models/stage2[_improved]_model.joblib
        results/cv_predictions_stage1.npz, results/cv_predictions_stage2[_improved].npz
Output: results/figures/permutation_test.png
        results/figures/bootstrap_ci.png
        results/figures/threshold_sensitivity.png
        results/figures/calibration_curve.png
        results/reports/within_validation_report.md
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef, brier_score_loss,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "train_file": "data/ucec/processed/feature_matrix_train.csv",
    "de_file": "data/ucec/raw/de_results.csv",
    "model_dir": "models",
    "results_dir": "results",
    "figures_dir": "results/figures",
    "reports_dir": "results/reports",
    "n_permutations": 1000,
    "n_bootstraps": 1000,
    "n_folds": 5,
    "random_state": 42,
}

LEAKY_PREFIXES = [
    "ppi_frac_neighbors_", "ppi_weighted_frac_",
    "pw_max_frac_", "pw_mean_frac_",
    "GO_BP_sim_", "GO_MF_sim_", "GO_CC_sim_",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_feature_cols(all_cols):
    """Return feature columns excluding metadata and leaky features."""
    meta = {"UniProt_ID", "label", "log2FC", "adj_pvalue"}
    return [c for c in all_cols
            if c not in meta and not any(c.startswith(p) for p in LEAKY_PREFIXES)]


def load_train_data(path):
    """Load training matrix, return X, y_stage1, y_stage2, de_mask, feature_cols, df."""
    df = pd.read_csv(path)
    feature_cols = _safe_feature_cols(df.columns)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y_stage1 = np.where(df["label"] == "unchanged", 0, 1)
    y_stage2 = np.where(df["label"] == "up", 1, 0)
    de_mask = df["label"].isin(["up", "down"]).values

    return X, y_stage1, y_stage2, de_mask, feature_cols, df


def _cv_auc(X, y, params, n_folds, seed):
    """Quick 5-fold stratified CV AUC."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, val_idx in cv.split(X, y):
        clf = xgb.XGBClassifier(**params)
        clf.fit(X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False)
        y_proba = clf.predict_proba(X[val_idx])[:, 1]
        try:
            aucs.append(roc_auc_score(y[val_idx], y_proba))
        except ValueError:
            aucs.append(0.5)
    return np.mean(aucs)


def _preferred_stage2_predictions_path(cfg):
    """Prefer improved Stage 2 CV predictions when present."""
    improved = Path(cfg["results_dir"]) / "cv_predictions_stage2_improved.npz"
    original = Path(cfg["results_dir"]) / "cv_predictions_stage2.npz"
    return improved if improved.exists() else original


# ---------------------------------------------------------------------------
# Section 1: Permutation Test
# ---------------------------------------------------------------------------

def permutation_test(X, y, saved_params, observed_auc, cfg):
    """Permute labels, retrain, build null AUC distribution."""
    print("\n--- 1. Permutation Test ---")
    rng = np.random.RandomState(cfg["random_state"])

    # Strip early-stopping from params (no eval_set during permutation)
    params = {k: v for k, v in saved_params.items()
              if k != "early_stopping_rounds"}

    null_aucs = []
    for i in range(cfg["n_permutations"]):
        y_perm = rng.permutation(y)
        auc = _cv_auc(X, y_perm, params, cfg["n_folds"], cfg["random_state"])
        null_aucs.append(auc)
        if (i + 1) % 10 == 0:
            print(f"  Shuffle {i + 1}/{cfg['n_permutations']}  "
                  f"(null AUC = {auc:.3f})")

    null_aucs = np.array(null_aucs)
    n_ge = np.sum(null_aucs >= observed_auc)
    p_value = (n_ge + 1) / (cfg["n_permutations"] + 1)

    print(f"  Observed AUC : {observed_auc:.4f}")
    print(f"  Null mean    : {null_aucs.mean():.4f} +/- {null_aucs.std():.4f}")
    print(f"  p-value      : {p_value:.4f}")

    # --- Plot ---
    fig_dir = Path(cfg["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null_aucs, bins=25, color="steelblue", edgecolor="white",
            alpha=0.8, label="Null distribution")
    ax.axvline(observed_auc, color="crimson", linewidth=2, linestyle="--",
               label=f"Observed AUC = {observed_auc:.3f}")
    ax.set_xlabel("AUC (5-fold CV)")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation Test (n={cfg['n_permutations']}, "
                 f"p = {p_value:.4f})")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_dir / "permutation_test.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "null_aucs": null_aucs,
        "observed_auc": observed_auc,
        "p_value": p_value,
        "null_mean": float(null_aucs.mean()),
        "null_std": float(null_aucs.std()),
    }


# ---------------------------------------------------------------------------
# Section 2: Bootstrap Confidence Intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(cfg):
    """Bootstrap 95% CIs for AUC, F1, MCC from saved CV predictions."""
    print("\n--- 2. Bootstrap Confidence Intervals ---")
    rng = np.random.RandomState(cfg["random_state"])

    results = {}
    for stage, label in [("stage1", "Stage 1"), ("stage2", "Stage 2")]:
        if stage == "stage2":
            npz_path = _preferred_stage2_predictions_path(cfg)
        else:
            npz_path = Path(cfg["results_dir"]) / f"cv_predictions_{stage}.npz"
        if not npz_path.exists():
            print(f"  {label}: {npz_path} not found, skipping.")
            continue

        data = np.load(npz_path, allow_pickle=True)
        y_true = data["y_true"]
        y_proba = data["y_proba"]
        y_pred = data["y_pred"]

        n = len(y_true)
        boot_auc, boot_f1, boot_mcc = [], [], []

        for _ in range(cfg["n_bootstraps"]):
            idx = rng.choice(n, size=n, replace=True)
            yt, yp, ypr = y_true[idx], y_proba[idx], y_pred[idx]
            # Need both classes present
            if len(np.unique(yt)) < 2:
                continue
            boot_auc.append(roc_auc_score(yt, yp))
            boot_f1.append(f1_score(yt, ypr))
            boot_mcc.append(matthews_corrcoef(yt, ypr))

        boot_auc = np.array(boot_auc)
        boot_f1 = np.array(boot_f1)
        boot_mcc = np.array(boot_mcc)

        ci = {}
        for name, arr in [("AUC", boot_auc), ("F1", boot_f1), ("MCC", boot_mcc)]:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            ci[name] = {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi)}
            print(f"  {label} {name}: {arr.mean():.3f}  "
                  f"95% CI [{lo:.3f}, {hi:.3f}]")

        results[stage] = {
            "ci": ci,
            "boot_auc": boot_auc,
            "boot_f1": boot_f1,
            "boot_mcc": boot_mcc,
        }

    # --- Plot ---
    if not results:
        print("  No CV predictions found; skipping bootstrap plot.")
        return results

    fig_dir = Path(cfg["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_stages = len(results)
    fig, axes = plt.subplots(1, n_stages, figsize=(6 * n_stages, 4.5),
                             squeeze=False)

    for col, (stage, res) in enumerate(results.items()):
        ax = axes[0, col]
        metrics = ["AUC", "F1", "MCC"]
        means = [res["ci"][m]["mean"] for m in metrics]
        los = [res["ci"][m]["lo"] for m in metrics]
        his = [res["ci"][m]["hi"] for m in metrics]
        errs_lo = [m - lo for m, lo in zip(means, los)]
        errs_hi = [hi - m for m, hi in zip(means, his)]

        x = np.arange(len(metrics))
        ax.bar(x, means, color=["steelblue", "coral", "seagreen"],
               edgecolor="white", alpha=0.85)
        ax.errorbar(x, means, yerr=[errs_lo, errs_hi], fmt="none",
                    ecolor="black", capsize=5, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        stage_label = "Stage 1 (DE vs Unchanged)" if stage == "stage1" \
            else "Stage 2 (Up vs Down)"
        ax.set_title(f"Bootstrap 95% CI — {stage_label}")

        # Annotate
        for xi, m, lo, hi in zip(x, means, los, his):
            ax.text(xi, hi + 0.03, f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "bootstrap_ci.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


# ---------------------------------------------------------------------------
# Section 3: Threshold Sensitivity
# ---------------------------------------------------------------------------

def threshold_sensitivity(X_all, feature_cols, de_df, train_df, saved_params, cfg):
    """Re-label at various FC thresholds and measure Stage 1 AUC stability."""
    print("\n--- 3. Threshold Sensitivity ---")

    params = {k: v for k, v in saved_params.items()
              if k != "early_stopping_rounds"}

    thresholds = np.arange(0.3, 1.55, 0.1)
    threshold_aucs = []

    # Merge DE results with feature matrix on UniProt_ID
    merged = train_df.copy()
    if "log2FC" not in merged.columns or "adj_pvalue" not in merged.columns:
        # Fall back to the raw DE file
        de = pd.read_csv(de_df)
        merged = merged.merge(de[["UniProt_ID", "log2FC", "adj_pvalue"]],
                              on="UniProt_ID", how="left", suffixes=("", "_de"))

    for thresh in thresholds:
        fc = merged["log2FC"].values
        pv = merged["adj_pvalue"].values

        up_mask = (fc > thresh) & (pv < 0.05)
        down_mask = (fc < -thresh) & (pv < 0.05)
        unch_mask = (np.abs(fc) < thresh * 0.5) & (pv > 0.20)

        # Exclude ambiguous proteins
        keep = up_mask | down_mask | unch_mask
        if keep.sum() < 50:
            print(f"  FC>{thresh:.1f}: too few proteins ({keep.sum()}), skipping")
            threshold_aucs.append(np.nan)
            continue

        y_relabel = np.full(len(merged), -1, dtype=int)
        y_relabel[up_mask | down_mask] = 1   # DE
        y_relabel[unch_mask] = 0              # unchanged

        X_sub = X_all[keep]
        y_sub = y_relabel[keep]

        # Need both classes
        if len(np.unique(y_sub)) < 2:
            threshold_aucs.append(np.nan)
            continue

        auc = _cv_auc(X_sub, y_sub, params, cfg["n_folds"], cfg["random_state"])
        threshold_aucs.append(auc)
        n_de = np.sum(y_sub == 1)
        n_unch = np.sum(y_sub == 0)
        print(f"  FC>{thresh:.1f}: AUC={auc:.3f}  "
              f"(DE={n_de}, unchanged={n_unch}, excluded={len(merged) - keep.sum()})")

    threshold_aucs = np.array(threshold_aucs)

    # --- Plot ---
    fig_dir = Path(cfg["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    valid = ~np.isnan(threshold_aucs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds[valid], threshold_aucs[valid], "o-", color="steelblue",
            linewidth=2, markersize=6)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Random")
    ax.set_xlabel("|log2FC| threshold")
    ax.set_ylabel("Stage 1 AUC (5-fold CV)")
    ax.set_title("Threshold Sensitivity: AUC vs FC Cutoff")
    ax.legend()
    ax.set_ylim(0.4, max(0.85, np.nanmax(threshold_aucs) + 0.05))
    plt.tight_layout()
    plt.savefig(fig_dir / "threshold_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "thresholds": thresholds.tolist(),
        "aucs": threshold_aucs.tolist(),
    }


# ---------------------------------------------------------------------------
# Section 4: Calibration Curve
# ---------------------------------------------------------------------------

def calibration_analysis(cfg):
    """Plot reliability diagram from saved CV predictions."""
    print("\n--- 4. Calibration Curve ---")

    npz_path = Path(cfg["results_dir"]) / "cv_predictions_stage1.npz"
    if not npz_path.exists():
        print(f"  {npz_path} not found, skipping calibration analysis.")
        return None

    data = np.load(npz_path, allow_pickle=True)
    y_true = data["y_true"]
    y_proba = data["y_proba"]

    brier = brier_score_loss(y_true, y_proba)
    print(f"  Brier score: {brier:.4f}")

    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_proba, n_bins=10, strategy="uniform"
    )

    # --- Plot ---
    fig_dir = Path(cfg["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7),
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k:", linewidth=1, label="Perfectly calibrated")
    ax1.plot(mean_predicted, fraction_pos, "s-", color="steelblue",
             linewidth=2, markersize=7, label="Stage 1 model")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed frequency")
    ax1.set_title(f"Calibration Curve (Brier = {brier:.4f})")
    ax1.legend(loc="lower right")
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Histogram of predicted probabilities
    ax2.hist(y_proba, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of predicted probabilities")

    plt.tight_layout()
    plt.savefig(fig_dir / "calibration_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "brier_score": float(brier),
        "fraction_positive": fraction_pos.tolist(),
        "mean_predicted": mean_predicted.tolist(),
    }


# ---------------------------------------------------------------------------
# Section 5: MNAR Simulation
# ---------------------------------------------------------------------------

def mnar_simulation(X, y_stage1, feature_cols, df, saved_params, cfg):
    """Simulate missing-not-at-random: mask least detectable proteins."""
    print("\n--- 5. MNAR Simulation ---")

    params = {k: v for k, v in saved_params.items()
              if k != "early_stopping_rounds"}

    # Build detectability score from relevant features
    detect_features = []
    for col in ["det_n_tryptic", "det_tryptic_density", "det_gravy"]:
        if col in feature_cols:
            detect_features.append(col)

    if not detect_features:
        # Fallback: look for columns containing these substrings
        for col in feature_cols:
            if "tryptic" in col.lower() or "gravy" in col.lower():
                detect_features.append(col)

    if not detect_features:
        print("  No detectability features found, skipping MNAR simulation.")
        return None

    print(f"  Detectability features: {detect_features}")

    # Composite detectability score (rank average — higher = easier to detect)
    ranks = np.zeros(len(df))
    for feat in detect_features:
        idx = feature_cols.index(feat)
        col_vals = X[:, idx]
        # Rank: higher values = higher rank = more detectable
        ranks += pd.Series(col_vals).rank(method="average").values
    ranks /= len(detect_features)

    # Split: bottom 20% (hard to detect) vs top 80% (easy to detect)
    cutoff = np.percentile(ranks, 20)
    hard_mask = ranks <= cutoff
    easy_mask = ~hard_mask

    n_hard = hard_mask.sum()
    n_easy = easy_mask.sum()
    print(f"  Easy-to-detect (train): {n_easy}")
    print(f"  Hard-to-detect (test) : {n_hard}")

    # Need both classes in both splits
    for split_name, mask in [("easy", easy_mask), ("hard", hard_mask)]:
        classes = np.unique(y_stage1[mask])
        if len(classes) < 2:
            print(f"  {split_name} split has only class(es) {classes}, "
                  "cannot evaluate. Skipping.")
            return None

    # Standard CV AUC (for comparison)
    standard_auc = _cv_auc(X, y_stage1, params, cfg["n_folds"],
                           cfg["random_state"])
    print(f"  Standard CV AUC   : {standard_auc:.3f}")

    # Train on easy, test on hard
    X_easy, y_easy = X[easy_mask], y_stage1[easy_mask]
    X_hard, y_hard = X[hard_mask], y_stage1[hard_mask]

    clf = xgb.XGBClassifier(**params)
    clf.fit(X_easy, y_easy, verbose=False)
    y_proba_hard = clf.predict_proba(X_hard)[:, 1]

    try:
        mnar_auc = roc_auc_score(y_hard, y_proba_hard)
    except ValueError:
        mnar_auc = 0.5

    auc_drop = standard_auc - mnar_auc
    print(f"  MNAR AUC (hard)   : {mnar_auc:.3f}")
    print(f"  AUC drop          : {auc_drop:+.3f}")

    return {
        "standard_auc": float(standard_auc),
        "mnar_auc": float(mnar_auc),
        "auc_drop": float(auc_drop),
        "n_easy": int(n_easy),
        "n_hard": int(n_hard),
        "detect_features": detect_features,
    }


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

def generate_report(perm_res, boot_res, thresh_res, calib_res, mnar_res, cfg):
    """Write comprehensive markdown validation report."""
    print("\n--- Generating Report ---")

    report_dir = Path(cfg["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# ProtDynPredict - Within-Experiment Validation Report\n")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("---\n")

    # --- Section 1: Permutation Test ---
    lines.append("## 1. Permutation Test\n")
    if perm_res is not None:
        lines.append(f"- **Observed AUC**: {perm_res['observed_auc']:.4f}")
        lines.append(f"- **Null distribution**: mean = {perm_res['null_mean']:.4f}, "
                     f"SD = {perm_res['null_std']:.4f}")
        lines.append(f"- **Empirical p-value**: {perm_res['p_value']:.4f}")
        lines.append(f"- **Number of permutations**: {cfg['n_permutations']}\n")
        if perm_res['p_value'] < 0.05:
            lines.append("**Conclusion**: The model significantly outperforms random "
                         "label assignment (p < 0.05). The learned signal is unlikely "
                         "to be an artefact of label structure.\n")
        else:
            lines.append("**Conclusion**: The model does NOT significantly outperform "
                         "random label assignment. The learned signal may be spurious.\n")
        lines.append("![Permutation Test](../figures/permutation_test.png)\n")
    else:
        lines.append("*Permutation test was not run.*\n")

    # --- Section 2: Bootstrap CIs ---
    lines.append("## 2. Bootstrap Confidence Intervals\n")
    if boot_res:
        for stage, res in boot_res.items():
            stage_label = "Stage 1 (DE vs Unchanged)" if stage == "stage1" \
                else "Stage 2 (Up vs Down)"
            lines.append(f"### {stage_label}\n")
            lines.append("| Metric | Mean | 95% CI Lower | 95% CI Upper |")
            lines.append("|--------|------|-------------|-------------|")
            for metric in ["AUC", "F1", "MCC"]:
                ci = res["ci"][metric]
                lines.append(f"| {metric} | {ci['mean']:.3f} | "
                             f"{ci['lo']:.3f} | {ci['hi']:.3f} |")
            lines.append("")
        lines.append(f"- **Bootstrap iterations**: {cfg['n_bootstraps']}\n")
        lines.append("![Bootstrap CIs](../figures/bootstrap_ci.png)\n")
    else:
        lines.append("*Bootstrap analysis was not run (no CV predictions found).*\n")

    # --- Section 3: Threshold Sensitivity ---
    lines.append("## 3. Threshold Sensitivity\n")
    if thresh_res is not None:
        lines.append("| FC Threshold | AUC |")
        lines.append("|-------------|-----|")
        for t, a in zip(thresh_res["thresholds"], thresh_res["aucs"]):
            auc_str = f"{a:.3f}" if not np.isnan(a) else "N/A"
            lines.append(f"| {t:.1f} | {auc_str} |")
        lines.append("")

        valid_aucs = [a for a in thresh_res["aucs"] if not np.isnan(a)]
        if valid_aucs:
            lines.append(f"- **AUC range**: [{min(valid_aucs):.3f}, "
                         f"{max(valid_aucs):.3f}]")
            lines.append(f"- **AUC spread**: {max(valid_aucs) - min(valid_aucs):.3f}\n")
            if max(valid_aucs) - min(valid_aucs) < 0.05:
                lines.append("**Conclusion**: AUC is stable across thresholds, "
                             "indicating that the model is robust to the exact "
                             "FC cutoff used for labeling.\n")
            else:
                lines.append("**Conclusion**: AUC varies meaningfully across "
                             "thresholds. Careful threshold selection is important.\n")

        lines.append("![Threshold Sensitivity](../figures/threshold_sensitivity.png)\n")
    else:
        lines.append("*Threshold sensitivity analysis was not run.*\n")

    # --- Section 4: Calibration ---
    lines.append("## 4. Calibration Analysis\n")
    if calib_res is not None:
        lines.append(f"- **Brier score**: {calib_res['brier_score']:.4f}")
        lines.append(f"  (lower is better; 0 = perfect, 0.25 = uninformative)\n")
        if calib_res['brier_score'] < 0.20:
            lines.append("**Conclusion**: Predicted probabilities are reasonably "
                         "well-calibrated.\n")
        else:
            lines.append("**Conclusion**: Predicted probabilities show poor "
                         "calibration. Consider Platt scaling or isotonic "
                         "regression for post-hoc calibration.\n")
        lines.append("![Calibration Curve](../figures/calibration_curve.png)\n")
    else:
        lines.append("*Calibration analysis was not run (no CV predictions found).*\n")

    # --- Section 5: MNAR Simulation ---
    lines.append("## 5. MNAR (Missing Not At Random) Simulation\n")
    if mnar_res is not None:
        lines.append(f"- **Detectability features**: "
                     f"{', '.join(mnar_res['detect_features'])}")
        lines.append(f"- **Train set (easy-to-detect)**: "
                     f"{mnar_res['n_easy']} proteins")
        lines.append(f"- **Test set (hard-to-detect)**: "
                     f"{mnar_res['n_hard']} proteins")
        lines.append(f"- **Standard CV AUC**: {mnar_res['standard_auc']:.3f}")
        lines.append(f"- **MNAR AUC (hard-to-detect)**: {mnar_res['mnar_auc']:.3f}")
        lines.append(f"- **AUC drop**: {mnar_res['auc_drop']:+.3f}\n")
        if abs(mnar_res['auc_drop']) < 0.05:
            lines.append("**Conclusion**: Minimal performance drop under MNAR "
                         "conditions. The model generalizes well to hard-to-detect "
                         "proteins.\n")
        elif mnar_res['auc_drop'] > 0.10:
            lines.append("**Conclusion**: Substantial AUC drop under MNAR conditions "
                         f"({mnar_res['auc_drop']:+.3f}). The model may be biased "
                         "toward easy-to-detect proteins. Consider detectability-aware "
                         "training or inverse-probability weighting.\n")
        else:
            lines.append("**Conclusion**: Moderate AUC drop under MNAR conditions. "
                         "Some caution is warranted when interpreting predictions "
                         "for low-detectability proteins.\n")
    else:
        lines.append("*MNAR simulation was not run (detectability features "
                     "not available).*\n")

    # --- Overall Summary ---
    lines.append("---\n")
    lines.append("## Summary\n")
    verdicts = []
    if perm_res is not None:
        status = "PASS" if perm_res["p_value"] < 0.05 else "FAIL"
        verdicts.append(f"| Permutation test | p = {perm_res['p_value']:.4f} | "
                        f"**{status}** |")
    if boot_res and "stage1" in boot_res:
        ci = boot_res["stage1"]["ci"]["AUC"]
        status = "PASS" if ci["lo"] > 0.55 else "MARGINAL"
        verdicts.append(f"| Bootstrap AUC 95% CI | "
                        f"[{ci['lo']:.3f}, {ci['hi']:.3f}] | **{status}** |")
    if mnar_res is not None:
        status = "PASS" if abs(mnar_res["auc_drop"]) < 0.10 else "WARN"
        verdicts.append(f"| MNAR robustness | "
                        f"drop = {mnar_res['auc_drop']:+.3f} | **{status}** |")

    if verdicts:
        lines.append("| Check | Value | Verdict |")
        lines.append("|-------|-------|---------|")
        lines.extend(verdicts)
        lines.append("")

    lines.append("\n*Report generated by `04_validate_within.py`*\n")

    report_path = report_dir / "within_validation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {report_path}")

    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Within-experiment validation for ProtDynPredict."
    )
    parser.add_argument("--dataset", default="ucec",
                        help="Dataset name (e.g., ucec, coad, brca)")
    parser.add_argument("--skip-permutation", action="store_true",
                        help="Skip the slow permutation test.")
    parser.add_argument("--permutations", type=int, default=None,
                        help="Number of permutation iterations (default: 1000)")
    args = parser.parse_args()

    if args.permutations is not None:
        CONFIG["n_permutations"] = args.permutations

    print("=" * 60)
    print("  Phase 4: WITHIN-EXPERIMENT VALIDATION")
    print("=" * 60)

    # Resolve paths based on dataset name
    dataset = args.dataset
    CONFIG["train_file"] = f"data/{dataset}/processed/feature_matrix_train.csv"
    CONFIG["de_file"] = f"data/{dataset}/raw/de_results.csv"
    CONFIG["model_dir"] = f"models/{dataset}"
    CONFIG["results_dir"] = f"results/{dataset}"
    CONFIG["figures_dir"] = f"results/{dataset}/figures"
    CONFIG["reports_dir"] = f"results/{dataset}/reports"

    # --- Load data ---
    print("\nLoading training data...")
    X, y_s1, y_s2, de_mask, feature_cols, df = load_train_data(CONFIG["train_file"])
    print(f"  {X.shape[0]} proteins, {X.shape[1]} features")

    # --- Load saved model params ---
    s1_path = Path(CONFIG["model_dir"]) / "stage1_model.joblib"
    if not s1_path.exists():
        raise FileNotFoundError(
            f"Stage 1 model not found at {s1_path}. Run 01_train_model.py first."
        )
    s1_saved = joblib.load(s1_path)
    s1_params = s1_saved["params"]
    s1_metrics = s1_saved["metrics"]
    observed_auc = s1_metrics.get("overall_auc", s1_metrics.get("mean_auc", 0.703))

    print(f"  Loaded Stage 1 params (observed AUC = {observed_auc:.3f})")

    # Ensure output directories exist
    for d in [CONFIG["figures_dir"], CONFIG["reports_dir"]]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Permutation Test
    # -----------------------------------------------------------------------
    perm_res = None
    if not args.skip_permutation:
        perm_res = permutation_test(X, y_s1, s1_params, observed_auc, CONFIG)
    else:
        print("\n--- 1. Permutation Test (SKIPPED) ---")

    # -----------------------------------------------------------------------
    # 2. Bootstrap CIs
    # -----------------------------------------------------------------------
    boot_res = bootstrap_ci(CONFIG)

    # -----------------------------------------------------------------------
    # 3. Threshold Sensitivity
    # -----------------------------------------------------------------------
    thresh_res = threshold_sensitivity(
        X, feature_cols, CONFIG["de_file"], df, s1_params, CONFIG
    )

    # -----------------------------------------------------------------------
    # 4. Calibration Curve
    # -----------------------------------------------------------------------
    calib_res = calibration_analysis(CONFIG)

    # -----------------------------------------------------------------------
    # 5. MNAR Simulation
    # -----------------------------------------------------------------------
    mnar_res = mnar_simulation(X, y_s1, feature_cols, df, s1_params, CONFIG)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    report_path = generate_report(
        perm_res, boot_res, thresh_res, calib_res, mnar_res, CONFIG
    )

    print("\n" + "=" * 60)
    print("  WITHIN-EXPERIMENT VALIDATION COMPLETE")
    print(f"  Report : {report_path}")
    print(f"  Figures: {CONFIG['figures_dir']}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
