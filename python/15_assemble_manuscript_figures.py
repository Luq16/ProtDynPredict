#!/usr/bin/env python3
"""
15_assemble_manuscript_figures.py

Create explicit numbered manuscript figures from the current regenerated
analysis outputs so the manuscript legends map to concrete files on disk.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score, roc_curve, auc


BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "ucec" / "processed"
RESULTS = BASE / "results" / "ucec"
FIGURES = RESULTS / "figures"
OUTDIR = RESULTS / "manuscript_figures"


def load_preferred_stage2_predictions():
    improved = RESULTS / "cv_predictions_stage2_improved.npz"
    original = RESULTS / "cv_predictions_stage2.npz"
    path = improved if improved.exists() else original
    return path, np.load(path)


def load_png(ax, path: Path, title: str | None = None):
    img = mpimg.imread(str(path))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")


def panel_label(ax, label: str):
    ax.text(
        0.0,
        1.02,
        f"({label})",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
    )


def save_fig(fig, name: str):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def make_figure_1():
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1], height_ratios=[1, 1], hspace=0.25, wspace=0.2)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    for ax in (ax_a, ax_b, ax_c):
        ax.set_axis_off()

    panel_label(ax_a, "A")
    panel_label(ax_b, "B")
    panel_label(ax_c, "C")

    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)

    stage2_path, stage2_preds = load_preferred_stage2_predictions()
    stage2_auc = roc_auc_score(stage2_preds["y_true"], stage2_preds["y_proba"])

    boxes = [
        (0.6, 7.8, 3.2, 1.2, "CPTAC UCEC proteomics\n3,117 proteins"),
        (0.6, 5.8, 3.2, 1.2, "Feature engineering\n1,176 raw -> 1,163 safe"),
        (0.6, 3.8, 3.2, 1.2, "Validation suite\nwithin, cross-dataset,\northogonal"),
        (5.2, 6.8, 3.6, 1.3, "Stage 1\nDE vs unchanged\nAUC 0.7088"),
        (5.2, 4.6, 3.6, 1.3, f"Stage 2\nup vs down\nAUC {stage2_auc:.4f}"),
        (5.2, 2.3, 3.6, 1.3, "Interpretation\nSHAP, ablation,\npathway/RNA checks"),
    ]
    for x, y, w, h, txt in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                               ec="#1f1f1f", fc="#f6f2e8", lw=1.5)
        ax_a.add_patch(patch)
        ax_a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=12)

    arrows = [
        ((3.8, 8.4), (5.1, 7.45)),
        ((3.8, 6.4), (5.1, 7.2)),
        ((6.95, 6.75), (6.95, 5.9)),
        ((6.95, 4.55), (6.95, 3.65)),
        ((3.8, 4.4), (5.1, 2.95)),
    ]
    for start, end in arrows:
        ax_a.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=16, lw=1.5, color="#333333"))
    ax_a.text(0, 9.6, "ProtDynPredict workflow", fontsize=16, fontweight="bold")

    counts = pd.Series({
        "Sequence": 999,
        "GO-slim": 150,
        "PPI": 3,
        "Detectability": 10,
        "Pathway": 1,
    })
    colors = ["#a34a28", "#3f7f6b", "#4d669b", "#c08d2b", "#7b5a9b"]
    ax_b.bar(counts.index, counts.values, color=colors)
    ax_b.set_title("Safe feature composition", fontsize=14)
    ax_b.set_ylabel("Number of features")
    ax_b.tick_params(axis="x", rotation=25)
    for i, v in enumerate(counts.values):
        ax_b.text(i, v + max(5, v * 0.01), str(v), ha="center", fontsize=11)
    ax_b.set_axis_on()

    ax_c.set_xlim(0, 10)
    ax_c.set_ylim(0, 6)
    boxes2 = [
        (0.8, 3.8, 3.0, 1.2, "All proteins"),
        (6.2, 3.8, 3.0, 1.2, "DE proteins"),
        (6.2, 1.4, 1.35, 1.2, "Up"),
        (7.85, 1.4, 1.35, 1.2, "Down"),
        (0.8, 1.4, 3.0, 1.2, "Unchanged"),
    ]
    for x, y, w, h, txt in boxes2:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08",
                               ec="#1f1f1f", fc="#eef3f7", lw=1.4)
        ax_c.add_patch(patch)
        ax_c.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=12)
    for start, end in [((3.85, 4.4), (6.1, 4.4)), ((3.85, 4.2), (3.85, 2.0)), ((7.7, 3.8), (7.2, 2.65)), ((7.7, 3.8), (8.5, 2.65))]:
        ax_c.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=16, lw=1.5, color="#333333"))
    ax_c.text(0.2, 5.4, "Two-stage classifier", fontsize=14, fontweight="bold")
    ax_c.set_axis_on()
    ax_c.set_xticks([])
    ax_c.set_yticks([])

    fig.suptitle("Figure 1. Study design and analytical framework", fontsize=18, fontweight="bold", y=0.98)
    save_fig(fig, "Figure_1.png")


def make_figure_2():
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.18, wspace=0.12)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    stage1_orig = np.load(RESULTS / "cv_predictions_stage1.npz")
    stage1_imp = np.load(RESULTS / "cv_predictions_stage1_improved.npz")
    stage2_orig = np.load(RESULTS / "cv_predictions_stage2.npz")
    _, stage2_imp = load_preferred_stage2_predictions()

    roc_specs = [
        ("Stage 1 improved", stage1_imp["y_true"], stage1_imp["y_proba"], "#b24c2f"),
        ("Stage 1 original", stage1_orig["y_true"], stage1_orig["y_proba"], "#7d8ea3"),
        ("Stage 2 improved", stage2_imp["y_true"], stage2_imp["y_proba"], "#2f6b59"),
        ("Stage 2 original", stage2_orig["y_true"], stage2_orig["y_proba"], "#8f7a48"),
    ]
    for name, y_true, y_prob, color in roc_specs:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_a.plot(fpr, tpr, lw=2.2, color=color, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax_a.plot([0, 1], [0, 1], "--", color="#666666", lw=1)
    ax_a.set_xlabel("False positive rate")
    ax_a.set_ylabel("True positive rate")
    ax_a.set_title("ROC curves from held-out CV predictions")
    ax_a.legend(frameon=False, loc="lower right")
    panel_label(ax_a, "A")

    load_png(ax_b, FIGURES / "baselines_comparison.png")
    panel_label(ax_b, "B")
    load_png(ax_c, FIGURES / "permutation_test.png")
    panel_label(ax_c, "C")
    load_png(ax_d, FIGURES / "threshold_sensitivity.png")
    panel_label(ax_d, "D")

    fig.suptitle("Figure 2. Within-dataset performance and robustness", fontsize=18, fontweight="bold", y=0.98)
    save_fig(fig, "Figure_2.png")


def make_figure_3():
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.15, wspace=0.12)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    load_png(ax_a, FIGURES / "shap_top30_stage_1_(de_vs_unchanged).png")
    panel_label(ax_a, "A")
    load_png(ax_b, FIGURES / "shap_categories_stage_1_(de_vs_unchanged).png")
    panel_label(ax_b, "B")
    load_png(ax_c, FIGURES / "feature_ablation.png")
    panel_label(ax_c, "C")

    fig.suptitle("Figure 3. Stage 1 interpretation and ablation", fontsize=18, fontweight="bold", y=0.98)
    save_fig(fig, "Figure_3.png")


def make_figure_3_stage2():
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.15, wspace=0.12)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    load_png(ax_a, FIGURES / "shap_top30_stage_2_(up_vs_down).png")
    panel_label(ax_a, "A")
    load_png(ax_b, FIGURES / "shap_categories_stage_2_(up_vs_down).png")
    panel_label(ax_b, "B")
    load_png(ax_c, FIGURES / "feature_ablation_stage2.png")
    panel_label(ax_c, "C")

    fig.suptitle("Stage 2 interpretation and ablation", fontsize=18, fontweight="bold", y=0.98)
    save_fig(fig, "Figure_3_stage2.png")


def make_figure_4():
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(1, 2, figure=fig, wspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    load_png(ax_a, FIGURES / "cross_dataset_auc.png")
    panel_label(ax_a, "A")
    load_png(ax_b, FIGURES / "cross_dataset_confusion.png")
    panel_label(ax_b, "B")

    fig.suptitle("Figure 4. Cross-dataset generalization", fontsize=18, fontweight="bold", y=0.98)
    save_fig(fig, "Figure_4.png")


def make_figure_5():
    fig = plt.figure(figsize=(16, 8.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.12)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    load_png(ax_a, FIGURES / "rna_protein_correlation.png")
    panel_label(ax_a, "A")
    load_png(ax_b, FIGURES / "pathway_enrichment.png")
    panel_label(ax_b, "B")

    fig.suptitle("Figure 5. Orthogonal biological validation", fontsize=18, fontweight="bold", y=0.98)
    save_fig(fig, "Figure_5.png")


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    ranked = pvals[order]
    n = len(pvals)
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = ranked[i] * n / (i + 1)
        prev = min(prev, val)
        adj[i] = prev
    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adj, 1.0)
    return out


def make_figure_6():
    train = pd.read_csv(DATA / "feature_matrix_train.csv")
    mapping = pd.read_csv(DATA / "go_slim_mapping.csv")
    _, preds = load_preferred_stage2_predictions()

    de = train[train["label"].isin(["up", "down"])].copy().reset_index(drop=True)
    de["pred_label"] = np.where(preds["y_pred"] == 1, "up", "down")
    cc = mapping[mapping["ONTOLOGY"] == "CC"].copy()

    true_rows = []
    for _, row in cc.iterrows():
        col = f"GOslim_{row['idx']}"
        if col not in de.columns:
            continue
        cnt_up = int(de.loc[de["label"] == "up", col].sum())
        cnt_down = int(de.loc[de["label"] == "down", col].sum())
        total = cnt_up + cnt_down
        if total < 20:
            continue
        table = [[cnt_up, (de["label"] == "up").sum() - cnt_up],
                 [cnt_down, (de["label"] == "down").sum() - cnt_down]]
        oratio, pval = fisher_exact(table)
        true_rows.append({
            "Compartment": row["TERM"],
            "Pct_Up": 100 * cnt_up / (de["label"] == "up").sum(),
            "Pct_Down": 100 * cnt_down / (de["label"] == "down").sum(),
            "OR": oratio,
            "p": pval,
        })
    tdf = pd.DataFrame(true_rows)
    tdf["p_adj"] = bh_adjust(tdf["p"].to_numpy())
    tdf["effect"] = np.abs(np.log(tdf["OR"].clip(lower=1e-8)))
    top_terms = tdf.sort_values(["p_adj", "effect"], ascending=[True, False]).head(15)["Compartment"].tolist()

    def pct_df(label_col: str, up_name: str, down_name: str):
        rows = []
        up_mask = de[label_col] == "up"
        down_mask = de[label_col] == "down"
        for _, row in cc.iterrows():
            col = f"GOslim_{row['idx']}"
            if col not in de.columns or row["TERM"] not in top_terms:
                continue
            cnt_up = int(de.loc[up_mask, col].sum())
            cnt_down = int(de.loc[down_mask, col].sum())
            rows.append({
                "Compartment": row["TERM"],
                up_name: 100 * cnt_up / up_mask.sum(),
                down_name: 100 * cnt_down / down_mask.sum(),
            })
        return pd.DataFrame(rows).set_index("Compartment").loc[top_terms].reset_index()

    true_plot = pct_df("label", "Up", "Down")
    pred_plot = pct_df("pred_label", "Pred up", "Pred down")

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(1, 2, figure=fig, wspace=0.15)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    y = np.arange(len(top_terms))
    h = 0.38

    ax_a.barh(y + h / 2, true_plot["Up"], h, color="#b24c2f", label="True up")
    ax_a.barh(y - h / 2, true_plot["Down"], h, color="#406e8e", label="True down")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(top_terms, fontsize=10)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("% annotated proteins")
    ax_a.set_title("True DE labels")
    ax_a.legend(frameon=False, loc="lower right")
    panel_label(ax_a, "A")

    ax_b.barh(y + h / 2, pred_plot["Pred up"], h, color="#cf7c4a", label="Predicted up")
    ax_b.barh(y - h / 2, pred_plot["Pred down"], h, color="#6c88a6", label="Predicted down")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(top_terms, fontsize=10)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("% annotated proteins")
    ax_b.set_title("Held-out Stage 2 predictions")
    ax_b.legend(frameon=False, loc="lower right")
    panel_label(ax_b, "B")

    fig.suptitle("Figure 6. Compartment-direction asymmetry in true and predicted labels", fontsize=18, fontweight="bold", y=0.98)
    save_fig(fig, "Figure_6.png")


def main():
    make_figure_1()
    make_figure_2()
    make_figure_3()
    make_figure_3_stage2()
    make_figure_4()
    make_figure_5()
    make_figure_6()


if __name__ == "__main__":
    main()
