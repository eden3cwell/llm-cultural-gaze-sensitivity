"""
RQ4 | Output 3: Architecture equity figures.

Produces:
    fig_normalised_curves.png      — normalised difficulty_delta polynomial curves
                                     for all conditions per model (4-col grid),
                                     fitted at the RQ1 Ferguson-retained degree
    fig_structural_shifts.png      — grouped bar chart of Δr+, Δr_peak, Δr- per
                                     condition across models
    fig_direction_heatmap.png      — % change in fitted amplitude: models ×
                                     conditions (both DVs)

Requires:
    outputs/new_experiment/RQ4/structural_point_shifts.csv
    outputs/new_experiment/RQ4/direction_table.csv

Run 01_structural_points.py and 02_direction_table.py first.

Output dir: outputs/new_experiment/RQ4/
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import (INFERENTIAL_MODELS, TREATMENT_CONDITIONS,
                    MODEL_LABELS, CONDITION_LABELS,
                    COL, OUTPUT_DIR, out_dir)
from loaders import load_all, cell_means

warnings.filterwarnings("ignore")

X_RANGE = np.linspace(0, 100, 1000)
DV = "difficulty_delta"

ALL_CONDITIONS = ["baseline"] + TREATMENT_CONDITIONS
COND_COLORS = {
    "baseline":             COL["baseline"],
    "maori":                COL["maori"],
    "kaumatua":             COL["kaumatua"],
    "gaze_aversion":        COL["gaze_aversion"],
    "direct_gaze_explicit": COL["direct_gaze_explicit"],
}


def load_rq1_degrees() -> dict:
    """Ferguson-retained degree per (model, dv) from RQ1. {} -> BIC fallback."""
    path = os.path.join(OUTPUT_DIR, "RQ1", "bic_polynomial_results.csv")
    if not os.path.exists(path):
        print("  [warn] RQ1 degrees not found; RQ4 falling back to local BIC")
        return {}
    rq1 = pd.read_csv(path)
    dv_map = {"Engagement Score": "engagement_score",
              "Difficulty Delta": "difficulty_delta"}
    col = "ferguson_degree" if "ferguson_degree" in rq1.columns else "selected_degree"
    return {(str(r["model"]).strip(),
             dv_map.get(str(r["dv"]).strip(), str(r["dv"]).strip())):
            int(r[col]) for _, r in rq1.iterrows()}


def bic_degree(x, y, max_degree=3):
    """Fallback degree selection (used only if Ferguson degrees unavailable)."""
    n = len(x)
    best_bic, best_deg = np.inf, 1
    for deg in range(1, max_degree + 1):
        c   = np.polyfit(x, y, deg)
        sse = float(np.sum((y - np.polyval(c, x)) ** 2))
        bic = n * np.log(sse / n) + (deg + 1) * np.log(n)
        if bic < best_bic:
            best_bic, best_deg = bic, deg
    return best_deg


def normalise(y_cond, y_bl_min, y_bl_max):
    span = y_bl_max - y_bl_min
    if span == 0:
        return np.zeros_like(y_cond)
    return 2 * (y_cond - y_bl_min) / span - 1


def plot_normalised_curves(dfs: dict) -> None:
    """4-column grid, all conditions overlaid, normalised difficulty_delta.
    Baseline spans [-1, +1] by construction; fitted at the Ferguson degree."""
    FERGUSON = load_rq1_degrees()
    models = [k for k in INFERENTIAL_MODELS if k in dfs]
    ncols  = 4
    nrows  = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes_flat = axes.flatten() if nrows > 1 else axes

    fig.suptitle(
        "Normalised Difficulty-Delta Polynomial Curves\n"
        "Baseline spans [−1, +1] by construction; treatment curves use same anchors",
        fontsize=11, fontweight="bold"
    )

    for ax_i, model_key in enumerate(models):
        ax    = axes_flat[ax_i]
        label = MODEL_LABELS[model_key]
        df_m  = dfs[model_key]

        bl  = df_m[df_m["condition"] == "baseline"]
        cm  = cell_means(bl, DV)
        x_r = cm["fixation_rate"].values.astype(float)
        y_r = cm[f"{DV}_mean"].values
        deg = FERGUSON.get((label, DV), bic_degree(x_r, y_r))

        bl_coeffs = np.polyfit(x_r, y_r, deg)
        y_bl      = np.polyval(bl_coeffs, X_RANGE)
        y_bl_min  = float(y_bl.min())
        y_bl_max  = float(y_bl.max())

        y_norm_bl = normalise(y_bl, y_bl_min, y_bl_max)
        ax.plot(X_RANGE, y_norm_bl,
                color=COND_COLORS["baseline"], lw=2.2, ls="--",
                label="Baseline", zorder=3)

        for cond in TREATMENT_CONDITIONS:
            tr = df_m[df_m["condition"] == cond]
            if len(tr) == 0:
                continue
            cm_c   = cell_means(tr, DV)
            x_c    = cm_c["fixation_rate"].values.astype(float)
            y_c    = cm_c[f"{DV}_mean"].values
            coeffs = np.polyfit(x_c, y_c, deg)
            y_fit  = np.polyval(coeffs, X_RANGE)
            y_norm = normalise(y_fit, y_bl_min, y_bl_max)
            ax.plot(X_RANGE, y_norm,
                    color=COND_COLORS[cond], lw=1.8,
                    label=CONDITION_LABELS.get(cond, cond).replace("\n", " "))

        ax.axhline(0,  color="#999999", lw=0.7, ls=":")
        ax.axhline(-1, color="#cccccc", lw=0.5, ls=":")
        ax.axhline(+1, color="#cccccc", lw=0.5, ls=":")
        ax.set_xlim(0, 100)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Fixation rate (%)", fontsize=8)
        ax.set_ylabel("Normalised δ", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax_i in range(len(models), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    handles, labels_ = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=6,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    path = os.path.join(out_dir("RQ4"), "fig_normalised_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_normalised_curves.png")


def plot_structural_shifts() -> None:
    """Grouped bar chart of Δr+, Δr_peak, Δr- per condition (x-axis = models)."""
    shifts_path = os.path.join(out_dir("RQ4"), "structural_point_shifts.csv")
    if not os.path.exists(shifts_path):
        print("  [SKIP] structural_point_shifts.csv not found"
              " — run 01_structural_points.py first")
        return

    df     = pd.read_csv(shifts_path)
    models = [MODEL_LABELS[k] for k in INFERENTIAL_MODELS
              if MODEL_LABELS[k] in df["model"].unique()]
    conds  = [c for c in TREATMENT_CONDITIONS if c in df["condition"].unique()]
    points = [("delta_r_plus", "Δr⁺ (positive crossing)"),
              ("delta_r_peak", "Δr_peak (maximum)"),
              ("delta_r_minus","Δr⁻ (negative crossing)")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Structural Point Shifts under Treatment Conditions\n"
        "(positive = shift right, negative = shift left; absent = monotone curve)",
        fontsize=11, fontweight="bold"
    )

    x = np.arange(len(models))
    w = 0.7 / max(len(conds), 1)

    for ax, (pt_col, pt_label) in zip(axes, points):
        for j, cond in enumerate(conds):
            sub  = df[df["condition"] == cond].set_index("model")
            vals = [
                float(sub.loc[m, pt_col]) if m in sub.index and not np.isnan(sub.loc[m, pt_col])
                else 0.0
                for m in models
            ]
            offset = (j - len(conds) / 2) * w + w / 2
            ax.bar(x + offset, vals, w * 0.9,
                   label=CONDITION_LABELS.get(cond, cond).replace("\n", " "),
                   color=COND_COLORS[cond], alpha=0.82)

        ax.axhline(0, color="#333333", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Shift in fixation rate (pp)", fontsize=9)
        ax.set_title(pt_label, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir("RQ4"), "fig_structural_shifts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_structural_shifts.png")


def plot_direction_heatmap() -> None:
    """Two heatmaps (one per DV): rows = models, cols = conditions.
    Cell = % change in fitted amplitude vs baseline (blue attenuate, red amplify)."""
    detail_path = os.path.join(out_dir("RQ4"), "direction_table.csv")
    if not os.path.exists(detail_path):
        print("  [SKIP] direction_table.csv not found — run 02_direction_table.py first")
        return

    df = pd.read_csv(detail_path)
    dvs = ["engagement_score", "difficulty_delta"]
    dv_labels = {"engagement_score": "Engagement Score",
                 "difficulty_delta":  "Difficulty Delta"}
    models = [MODEL_LABELS[k] for k in INFERENTIAL_MODELS
              if MODEL_LABELS[k] in df["model"].unique()]
    conds  = [c for c in TREATMENT_CONDITIONS if c in df["condition"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Change in Fitted Response Amplitude vs Baseline (%)\n"
        "Blue = attenuated / flattened heuristic; Red = amplified / steepened",
        fontsize=11, fontweight="bold"
    )

    cmap = plt.cm.RdBu_r
    vmax = 100.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, dv in zip(axes, dvs):
        sub = df[df["dv"] == dv]
        mat = np.full((len(models), len(conds)), np.nan)
        for i, m in enumerate(models):
            for j, c in enumerate(conds):
                cell = sub[(sub["model"] == m) & (sub["condition"] == c)]["pct_change"]
                if len(cell) > 0:
                    mat[i, j] = float(cell.iloc[0])

        im = ax.imshow(np.clip(mat, -vmax, vmax), cmap=cmap, norm=norm, aspect="auto")
        plt.colorbar(im, ax=ax, label="% change in amplitude", shrink=0.85)

        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(
            [CONDITION_LABELS.get(c, c).replace("\n", " ") for c in conds],
            rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(dv_labels[dv], fontsize=10, fontweight="bold")

        for i in range(len(models)):
            for j in range(len(conds)):
                v = mat[i, j]
                if not np.isnan(v):
                    tc = "white" if abs(np.clip(v, -vmax, vmax)) > vmax * 0.6 else "black"
                    ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                            fontsize=7.5, color=tc, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir("RQ4"), "fig_direction_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_direction_heatmap.png")


if __name__ == "__main__":
    print("RQ4 | Output 3: Figures\n" + "=" * 60)
    dfs = load_all()
    plot_normalised_curves(dfs)
    plot_structural_shifts()
    plot_direction_heatmap()