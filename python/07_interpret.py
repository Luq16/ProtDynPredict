#!/usr/bin/env python3
"""
07_interpret.py
SHAP-based model interpretation for both stages.

Key question: Do network features dominate?
  If yes → model is essentially doing sophisticated label propagation
  If sequence/GO features contribute → novel biological insight

Input:  models/stage1_model.joblib, models/stage2_model.joblib
        data/processed/feature_matrix_train.csv
Output: results/figures/shap_*.png
        results/reports/interpretation_report.md
"""

import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure the python/ directory is on the path for utils import
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.feature_decoder import decode_feature, get_property_group, PROPERTY_GROUPS

CONFIG = {
    "train_file": "data/ucec/processed/feature_matrix_train.csv",
    "model_dir": "models",
    "figures_dir": "results/figures",
    "reports_dir": "results/reports",
    "max_samples_shap": 500,
}


def categorize_feature(name):
    """Categorize a feature by its source."""
    if name.startswith(("AAC_", "DC_", "CTD", "PseAAC_", "APseAAC_",
                        "CTriad_", "QSO_", "SOCN_")):
        return "Sequence (protr)"
    elif name.startswith("GOslim_"):
        return "GO-slim binary"
    elif name.startswith("GO_"):
        return "GO group similarity"
    elif name.startswith("ppi_"):
        return "PPI network"
    elif name.startswith("pw_"):
        return "Pathway context"
    elif name.startswith("det_"):
        return "Detectability"
    else:
        return "Other"


def feature_property_enrichment(feature_importance, n_top=50):
    """Test whether property groups are enriched among top features.

    Uses Fisher's exact test to check if each property group is
    over-represented in the top-*n_top* features relative to all features.

    Returns a DataFrame with columns: group, top_count, total_count,
    odds_ratio, p_value.
    """
    all_features = feature_importance["feature"].tolist()
    top_features = feature_importance.head(n_top)["feature"].tolist()

    n_all = len(all_features)
    n_top_actual = len(top_features)

    results = []
    for group_name in list(PROPERTY_GROUPS.keys()) + ["other"]:
        # Count features belonging to this group across all and top
        in_group_all = sum(1 for f in all_features if group_name in get_property_group(f))
        in_group_top = sum(1 for f in top_features if group_name in get_property_group(f))

        not_in_group_all = n_all - in_group_all
        not_in_group_top = n_top_actual - in_group_top

        # 2x2 contingency table:
        #                in_top    not_in_top
        # in_group       a         b
        # not_in_group   c         d
        a = in_group_top
        b = in_group_all - in_group_top
        c = n_top_actual - in_group_top
        d = not_in_group_all - c

        table = [[a, b], [c, d]]
        try:
            odds_ratio, p_value = fisher_exact(table, alternative="greater")
        except Exception:
            odds_ratio, p_value = float("nan"), float("nan")

        results.append({
            "group": group_name,
            "top_count": in_group_top,
            "total_count": in_group_all,
            "odds_ratio": round(odds_ratio, 2) if not np.isnan(odds_ratio) else "inf",
            "p_value": round(p_value, 4),
        })

    return pd.DataFrame(results).sort_values("p_value")


def analyze_stage(model_path, X, feature_cols, stage_name, output_dir):
    """Run SHAP analysis for one model stage."""
    print(f"\n--- SHAP Analysis: {stage_name} ---")

    data = joblib.load(model_path)
    model = data["model"]

    # Subsample for speed
    n = min(CONFIG["max_samples_shap"], X.shape[0])
    idx = np.random.RandomState(42).choice(X.shape[0], n, replace=False)
    X_sample = X[idx]

    # SHAP values - use newer Explanation API for compatibility
    explainer = shap.TreeExplainer(model)
    try:
        explanation = explainer(X_sample)
        sv = explanation.values
        # For binary classification, sv may be 3D: (samples, features, classes)
        if sv.ndim == 3:
            sv = sv[:, :, 1]  # positive class
    except Exception:
        # Fallback to older API
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(sv).mean(axis=0)

    # Feature importance ranking
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
        "category": [categorize_feature(f) for f in feature_cols],
    }).sort_values("mean_abs_shap", ascending=False)

    # Category-level importance
    category_importance = feature_importance.groupby("category")["mean_abs_shap"].sum()
    category_importance = category_importance.sort_values(ascending=False)

    total_importance = category_importance.sum()
    category_pct = (category_importance / total_importance * 100).round(1)

    print("\n  Feature category contributions:")
    for cat, pct in category_pct.items():
        print(f"    {cat:<25} {pct:>5.1f}%")

    # --- Plots ---
    fig_dir = Path(output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    stage_tag = stage_name.lower().replace(" ", "_")

    # 1. SHAP summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X_sample, feature_names=feature_cols,
                      max_display=20, show=False)
    plt.title(f"SHAP Summary: {stage_name}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_summary_{stage_tag}.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # 2. Category-level bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "Sequence (protr)": "#4C72B0",
        "GO-slim binary": "#55A868",
        "GO group similarity": "#C44E52",
        "PPI network": "#8172B2",
        "Pathway context": "#CCB974",
        "Detectability": "#64B5CD",
        "Other": "#999999",
    }
    cats = category_pct.index.tolist()
    vals = category_pct.values
    bar_colors = [colors.get(c, "#999999") for c in cats]

    ax.barh(range(len(cats)), vals, color=bar_colors)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel("% of Total SHAP Importance")
    ax.set_title(f"Feature Category Contributions: {stage_name}")
    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_categories_{stage_tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Top N individual features (up to 30)
    top_n = min(30, len(feature_importance))
    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.3)))
    top30 = feature_importance.head(top_n)
    bar_colors = [colors.get(c, "#999999") for c in top30["category"]]
    ax.barh(range(top_n), top30["mean_abs_shap"].values[::-1], color=bar_colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top30["feature"].values[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Top 30 Features: {stage_name}")

    # Legend
    from matplotlib.patches import Patch
    legend_cats = top30["category"].unique()
    handles = [Patch(color=colors.get(c, "#999999"), label=c) for c in legend_cats]
    ax.legend(handles=handles, fontsize=7, loc="lower right")

    plt.tight_layout()
    plt.savefig(fig_dir / f"shap_top30_{stage_tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

    return feature_importance, category_pct


def main():
    parser = argparse.ArgumentParser(description="SHAP model interpretation")
    parser.add_argument("--dataset", default="ucec", help="Dataset/cancer type (default: ucec)")
    args = parser.parse_args()
    dataset = args.dataset
    CONFIG["train_file"] = f"data/{dataset}/processed/feature_matrix_train.csv"
    CONFIG["model_dir"] = f"models/{dataset}"
    CONFIG["figures_dir"] = f"results/{dataset}/figures"
    CONFIG["reports_dir"] = f"results/{dataset}/reports"

    print("=" * 60)
    print("  MODEL INTERPRETATION (SHAP)")
    print(f"  Dataset: {dataset}")
    print("=" * 60)

    # Load data
    df = pd.read_csv(CONFIG["train_file"])
    meta_cols = ["UniProt_ID", "label", "log2FC", "adj_pvalue"]
    all_feature_cols = [c for c in df.columns if c not in meta_cols]

    model_dir = Path(CONFIG["model_dir"])

    # Stage 1 - use the model's feature list to match
    s1_path = model_dir / "stage1_model.joblib"
    s1_importance = s1_categories = None
    if s1_path.exists():
        s1_data = joblib.load(s1_path)
        if "feature_cols" in s1_data:
            feature_cols = s1_data["feature_cols"]
        else:
            feature_cols = all_feature_cols
        X = df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        s1_importance, s1_categories = analyze_stage(
            s1_path, X, feature_cols, "Stage 1 (DE vs Unchanged)", CONFIG["figures_dir"]
        )

    # Stage 2 (DE proteins only)
    s2_path = model_dir / "stage2_model.joblib"
    s2_importance = s2_categories = None
    if s2_path.exists():
        s2_data = joblib.load(s2_path)
        if "feature_cols" in s2_data:
            feature_cols_s2 = s2_data["feature_cols"]
        else:
            feature_cols_s2 = all_feature_cols
        X_s2 = df[feature_cols_s2].values.astype(np.float32)
        X_s2 = np.nan_to_num(X_s2, nan=0.0, posinf=0.0, neginf=0.0)
        de_mask = df["label"].isin(["up", "down"]).values
        X_de = X_s2[de_mask]
        if len(X_de) >= 10:
            s2_importance, s2_categories = analyze_stage(
                s2_path, X_de, feature_cols_s2, "Stage 2 (Up vs Down)", CONFIG["figures_dir"]
            )

    # --- Generate report ---
    report_dir = Path(CONFIG["reports_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Model Interpretation Report\n",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
    ]

    for name, importance, categories in [
        ("Stage 1 (DE vs Unchanged)", s1_importance, s1_categories),
        ("Stage 2 (Up vs Down)", s2_importance, s2_categories),
    ]:
        if importance is None:
            continue

        lines.append(f"## {name}\n")
        lines.append("### Feature Category Contributions\n")
        lines.append("| Category | % SHAP Importance |")
        lines.append("|----------|-------------------|")
        for cat, pct in categories.items():
            lines.append(f"| {cat} | {pct:.1f}% |")

        lines.append(f"\n### Top 15 Features\n")
        lines.append("| Rank | Feature | Category | Mean |SHAP| |")
        lines.append("|------|---------|----------|------------|")
        for i, row in importance.head(15).iterrows():
            lines.append(f"| {importance.index.get_loc(i)+1} | `{row['feature']}` | "
                        f"{row['category']} | {row['mean_abs_shap']:.4f} |")
        lines.append("")

        # --- Biological Interpretation section ---
        lines.append(f"## Biological Interpretation: {name}\n")

        # Annotated Top 30 table
        lines.append("### Annotated Top 30 Features\n")
        lines.append("| Rank | Feature | Description | Category | Mean |SHAP| |")
        lines.append("|------|---------|-------------|----------|------------|")
        for idx_pos, (i, row) in enumerate(importance.head(30).iterrows(), 1):
            desc = decode_feature(row["feature"])
            lines.append(
                f"| {idx_pos} | `{row['feature']}` | {desc} | "
                f"{row['category']} | {row['mean_abs_shap']:.4f} |"
            )
        lines.append("")

        # Property enrichment analysis
        lines.append("### Feature Property Enrichment (Top 50 vs All)\n")
        enrichment_df = feature_property_enrichment(importance, n_top=50)
        lines.append("| Property Group | In Top 50 | Total | Odds Ratio | p-value |")
        lines.append("|---------------|-----------|-------|------------|---------|")
        for _, erow in enrichment_df.iterrows():
            lines.append(
                f"| {erow['group']} | {erow['top_count']} | {erow['total_count']} | "
                f"{erow['odds_ratio']} | {erow['p_value']:.4f} |"
            )
        lines.append("")

        # Narrative interpretation
        lines.append("### Narrative Interpretation\n")
        sig_groups = enrichment_df[enrichment_df["p_value"] < 0.05]
        if len(sig_groups) > 0:
            group_list = ", ".join(sig_groups["group"].tolist())
            lines.append(
                f"Significantly enriched property groups among the top 50 features "
                f"(Fisher's exact test, p < 0.05): **{group_list}**.\n"
            )
            for _, sg in sig_groups.iterrows():
                lines.append(
                    f"- **{sg['group']}**: {sg['top_count']} of {sg['total_count']} features "
                    f"in top 50 (OR={sg['odds_ratio']}, p={sg['p_value']:.4f})"
                )
            lines.append("")
            lines.append(
                "These enrichments suggest the model relies on specific physicochemical "
                "and biological properties to distinguish protein expression dynamics, "
                "rather than arbitrary feature combinations."
            )
        else:
            lines.append(
                "No property groups were significantly enriched among the top 50 features "
                "(Fisher's exact test, p < 0.05). This suggests the model draws on a diverse "
                "set of feature types without strong bias toward any single physicochemical property."
            )
        lines.append("")

    # Key insight
    if s1_categories is not None:
        network_pct = s1_categories.get("PPI network", 0) + s1_categories.get("Pathway context", 0)
        sequence_pct = s1_categories.get("Sequence (protr)", 0)
        go_pct = s1_categories.get("GO-slim binary", 0) + s1_categories.get("GO group similarity", 0)

        lines.append("## Key Insight\n")
        if network_pct > 50:
            lines.append("**Network features dominate.** The model is primarily leveraging "
                        "PPI network and pathway context features. This suggests the model "
                        "is doing sophisticated label propagation. The added value of "
                        "sequence/GO features is limited.")
        elif sequence_pct > 30:
            lines.append("**Sequence features contribute meaningfully.** This suggests "
                        "protein-intrinsic properties carry predictive signal for expression "
                        "dynamics — a novel biological finding worth investigating further.")
        else:
            lines.append(f"**Mixed signal.** Network: {network_pct:.0f}%, Sequence: {sequence_pct:.0f}%, "
                        f"GO: {go_pct:.0f}%. Multiple feature types contribute.")

    with open(report_dir / "interpretation_report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n  Report: {report_dir / 'interpretation_report.md'}")
    print("  Done.")


if __name__ == "__main__":
    main()
