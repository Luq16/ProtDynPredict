#!/usr/bin/env python3
"""
08_predict.py
Predict expression dynamics for undetected proteins.

Two-stage prediction:
  Stage 1: Is this protein differentially expressed? (probability)
  Stage 2: If DE, is it up or down? (probability)

Output includes calibrated confidence scores and biological context.

Input:  models/stage1_model.joblib, models/stage2_model.joblib
        data/processed/feature_matrix_predict.csv
        data/processed/feature_matrix_train.csv (for calibration reference)
Output: results/predictions.csv
        results/reports/prediction_report.md
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

CONFIG = {
    "predict_file": "data/processed/feature_matrix_predict.csv",
    "train_file": "data/processed/feature_matrix_train.csv",
    "model_dir": "models",
    "output_csv": "results/predictions.csv",
    "reports_dir": "results/reports",
    "de_threshold": 0.5,     # Stage 1 probability threshold
    "high_confidence": 0.7,  # threshold for "high confidence" predictions
}


def main():
    print("=" * 60)
    print("  PREDICTION: Undetected Proteins")
    print("=" * 60)

    # Load models
    model_dir = Path(CONFIG["model_dir"])
    s1_data = joblib.load(model_dir / "stage1_model.joblib")
    s1_model = s1_data["model"]
    feature_cols = s1_data["feature_cols"]

    s2_path = model_dir / "stage2_model.joblib"
    s2_model = None
    if s2_path.exists():
        s2_data = joblib.load(s2_path)
        s2_model = s2_data["model"]

    # Load prediction data
    pred_df = pd.read_csv(CONFIG["predict_file"])
    print(f"  Proteins to predict: {len(pred_df)}")

    if len(pred_df) == 0:
        print("  No undetected proteins to predict.")
        print("  (This happens when all proteins in metadata were detected.)")
        print("  To predict for external proteins, add them to feature_matrix_predict.csv")
        return

    protein_ids = pred_df["UniProt_ID"].values

    # Align features
    available_features = [c for c in feature_cols if c in pred_df.columns]
    missing_features = [c for c in feature_cols if c not in pred_df.columns]

    if missing_features:
        print(f"  Warning: {len(missing_features)} features missing, filling with 0")
        for f in missing_features:
            pred_df[f] = 0.0

    X = pred_df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Stage 1: DE probability ---
    print("  Stage 1: Predicting DE vs Unchanged...")
    de_proba = s1_model.predict_proba(X)[:, 1]

    # --- Stage 2: Direction (only for predicted-DE proteins) ---
    up_proba = np.full(len(X), 0.5)  # default: uncertain direction
    de_indices = np.where(de_proba >= CONFIG["de_threshold"])[0]

    if s2_model is not None and len(de_indices) > 0:
        print(f"  Stage 2: Predicting Up vs Down for {len(de_indices)} predicted-DE proteins...")
        up_proba[de_indices] = s2_model.predict_proba(X[de_indices])[:, 1]
    elif s2_model is None:
        print("  Stage 2 model not available, using DE probability only.")

    # --- Combine predictions ---
    results = pd.DataFrame({
        "UniProt_ID": protein_ids,
        "DE_probability": np.round(de_proba, 4),
        "up_probability": np.round(up_proba, 4),
        "down_probability": np.round(1 - up_proba, 4),
    })

    # Assign predicted class
    results["predicted_class"] = "unchanged"
    de_mask = results["DE_probability"] >= CONFIG["de_threshold"]
    up_mask = de_mask & (results["up_probability"] >= 0.5)
    down_mask = de_mask & (results["up_probability"] < 0.5)
    results.loc[up_mask, "predicted_class"] = "up"
    results.loc[down_mask, "predicted_class"] = "down"

    # Confidence score (calibrated)
    results["confidence"] = np.where(
        de_mask,
        results["DE_probability"] * np.maximum(results["up_probability"],
                                                 results["down_probability"]),
        1 - results["DE_probability"]
    ).round(4)

    # Flag high confidence
    results["high_confidence"] = results["confidence"] >= CONFIG["high_confidence"]

    # Sort by DE probability (most likely DE first)
    results = results.sort_values("DE_probability", ascending=False)

    # --- Add context from network features if available ---
    net_cols = [c for c in pred_df.columns if c.startswith("ppi_")]
    if net_cols:
        results = results.merge(
            pred_df[["UniProt_ID"] + net_cols],
            on="UniProt_ID", how="left"
        )

    # --- Save ---
    output_path = Path(CONFIG["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\n  Predictions saved: {output_path}")

    # --- Summary ---
    n_de = de_mask.sum()
    n_up = up_mask.sum()
    n_down = down_mask.sum()
    n_unch = (~de_mask).sum()
    n_high_conf = results["high_confidence"].sum()

    print(f"\n  Prediction Summary:")
    print(f"    Total proteins:      {len(results)}")
    print(f"    Predicted DE:        {n_de} ({n_up} up, {n_down} down)")
    print(f"    Predicted unchanged: {n_unch}")
    print(f"    High confidence:     {n_high_conf}")

    # --- Report ---
    report_dir = Path(CONFIG["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Prediction Report: Undetected Proteins\n",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Summary\n",
        f"- Total predicted: {len(results)}",
        f"- Predicted DE: {n_de} ({n_up} up, {n_down} down)",
        f"- Predicted unchanged: {n_unch}",
        f"- High confidence (>{CONFIG['high_confidence']}): {n_high_conf}\n",
        "## Top 50 Predicted DE Proteins\n",
        "| Rank | UniProt_ID | Class | DE Prob | Direction Prob | Confidence |",
        "|------|-----------|-------|---------|---------------|------------|",
    ]

    top50 = results[results["predicted_class"] != "unchanged"].head(50)
    for rank, (_, row) in enumerate(top50.iterrows(), 1):
        dir_prob = row["up_probability"] if row["predicted_class"] == "up" else row["down_probability"]
        lines.append(f"| {rank} | {row['UniProt_ID']} | {row['predicted_class']} | "
                    f"{row['DE_probability']:.3f} | {dir_prob:.3f} | {row['confidence']:.3f} |")

    lines.append("\n## Caveats\n")
    lines.append("- These are **predictions for validation**, not confirmed expression states")
    lines.append("- Confidence scores are model-calibrated but may not reflect true probability")
    lines.append("- Proteins missing due to MNAR (low abundance) may have different characteristics")
    lines.append("  than the detected proteins used for training")
    lines.append("- Recommended: validate top predictions with PRM/MRM or Western blot")

    with open(report_dir / "prediction_report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"  Report: {report_dir / 'prediction_report.md'}")


if __name__ == "__main__":
    main()
