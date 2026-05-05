#!/usr/bin/env python3
"""
12_supplementary_table_s4.py - Generate Table S4: Top 50 predicted DE proteins
with confidence scores from proteome-wide predictions.

For each dataset pair (train on A, predict on B), identifies proteins in B
that are absent from A ("undetected") and ranks them by predicted P(DE).
Outputs a consolidated supplementary table of the top 50 highest-confidence
predicted DE proteins across all pairs.

Output: results/ucec/reports/table_s4_top50_predicted_de.csv
        results/ucec/reports/table_s4_top50_predicted_de.md
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

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


def _is_leaky(col: str) -> bool:
    return any(col.startswith(p) for p in LEAKY_PREFIXES)


def safe_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS and not _is_leaky(c)]


def load_dataset(name: str) -> pd.DataFrame | None:
    path = Path("data") / name / "processed" / "feature_matrix_train.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found -- skipping {name}")
        return None
    return pd.read_csv(path)


def get_model_spec(dataset: str) -> tuple[dict, list[str] | None]:
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
                    feature_cols = None
                    if bundle.get("feature_type") == "non-sequence":
                        feature_cols = bundle.get("feature_cols") or bundle.get("nonseq_cols")
                    elif bundle.get("feature_type") == "full":
                        feature_cols = bundle.get("feature_cols")
                    return params, feature_cols
            except Exception:
                pass
    return dict(DEFAULT_XGB_PARAMS), None


def main():
    print("=" * 60)
    print("  Generating Table S4: Top 50 Predicted DE Proteins")
    print("=" * 60)

    # Load all datasets
    data = {}
    for ds in DATASETS:
        df = load_dataset(ds)
        if df is not None:
            data[ds] = df
            print(f"  {ds}: {len(df)} proteins")

    if len(data) < 2:
        print("ERROR: Need at least 2 datasets.")
        return

    all_predictions = []
    names = sorted(data.keys())

    for train_name in names:
        params, preferred_cols = get_model_spec(train_name)
        train_df = data[train_name]

        for test_name in names:
            if train_name == test_name:
                continue

            test_df = data[test_name]
            train_ids = set(train_df["UniProt_ID"])
            test_ids = set(test_df["UniProt_ID"])
            undetected_ids = test_ids - train_ids

            if not undetected_ids:
                continue

            # Feature alignment
            train_feats = set(safe_feature_cols(train_df))
            test_feats = set(safe_feature_cols(test_df))
            shared_cols = sorted(train_feats & test_feats)
            if preferred_cols is not None:
                shared_cols = [c for c in shared_cols if c in set(preferred_cols)]

            # Train
            X_train = train_df[shared_cols].values.astype(np.float32)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = np.where(train_df["label"].values == "unchanged", 0, 1)

            scale_pos = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
            run_params = dict(params)
            run_params["scale_pos_weight"] = scale_pos

            clf = xgb.XGBClassifier(**run_params)
            clf.fit(X_train, y_train, verbose=False)

            # Predict on undetected
            test_undet = test_df[test_df["UniProt_ID"].isin(undetected_ids)].copy()
            X_test = test_undet[shared_cols].values.astype(np.float32)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            y_proba = clf.predict_proba(X_test)[:, 1]
            test_undet = test_undet.copy()
            test_undet["P_DE"] = y_proba
            test_undet["train_dataset"] = train_name
            test_undet["test_dataset"] = test_name
            test_undet["true_label"] = test_undet["label"]

            all_predictions.append(
                test_undet[["UniProt_ID", "true_label", "log2FC", "adj_pvalue",
                            "P_DE", "train_dataset", "test_dataset"]]
            )

            print(f"  {train_name} -> {test_name}: "
                  f"{len(test_undet)} undetected proteins scored")

    # Combine and rank
    combined = pd.concat(all_predictions, ignore_index=True)

    # Average P(DE) across all pairs where this protein appears as undetected
    agg = combined.groupby("UniProt_ID").agg(
        mean_P_DE=("P_DE", "mean"),
        max_P_DE=("P_DE", "max"),
        n_pairs=("P_DE", "count"),
        true_labels=("true_label", lambda x: ",".join(sorted(set(x)))),
        mean_log2FC=("log2FC", "mean"),
        mean_adj_pvalue=("adj_pvalue", "mean"),
    ).reset_index()

    # Top 50 by mean P(DE)
    top50 = agg.nlargest(50, "mean_P_DE").reset_index(drop=True)
    top50.index = top50.index + 1  # 1-indexed rank
    top50.index.name = "Rank"

    # Output paths
    out_dir = Path("results/ucec/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "table_s4_top50_predicted_de.csv"
    top50.to_csv(csv_path)
    print(f"\n  CSV saved: {csv_path}")

    # Markdown
    md_path = out_dir / "table_s4_top50_predicted_de.md"
    lines = []
    lines.append("# Table S4: Top 50 Predicted Differentially Expressed Proteins\n")
    lines.append(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append("Proteins ranked by mean predicted P(DE) across all cross-dataset pairs")
    lines.append("where the protein was absent from the training dataset (\"undetected\").\n")
    lines.append("| Rank | UniProt_ID | Mean P(DE) | Max P(DE) | N_pairs | True Label(s) | Mean log2FC | Mean adj.p |")
    lines.append("|------|-----------|-----------|----------|---------|---------------|------------|-----------|")
    for rank, row in top50.iterrows():
        lines.append(
            f"| {rank} | {row['UniProt_ID']} | {row['mean_P_DE']:.3f} | "
            f"{row['max_P_DE']:.3f} | {row['n_pairs']} | {row['true_labels']} | "
            f"{row['mean_log2FC']:.3f} | {row['mean_adj_pvalue']:.2e} |"
        )

    lines.append(f"\n**Total proteins scored**: {len(agg)}")
    lines.append(f"**Mean P(DE) range in top 50**: "
                 f"{top50['mean_P_DE'].min():.3f} -- {top50['mean_P_DE'].max():.3f}")

    # Quick accuracy check
    n_true_de = top50["true_labels"].apply(
        lambda x: any(l in ("up", "down") for l in x.split(","))
    ).sum()
    lines.append(f"**True DE among top 50**: {n_true_de}/50 "
                 f"({n_true_de/50:.0%})")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Markdown saved: {md_path}")

    print("\n" + "=" * 60)
    print("  TABLE S4 GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
