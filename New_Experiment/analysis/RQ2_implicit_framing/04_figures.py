"""
RQ2 | Output 4: Figures — binned mean-difference heatmaps + JN cross-model.

Heatmaps now plot RAW MEAN DIFFERENCE per bin (DV units), not Cohen's d.
Cross-model JN overlays the polynomial floodlight per condition, per DV, at
the RQ1 Ferguson degree.

Produces:
    fig_meandiff_binned_engagement.png
    fig_meandiff_binned_delta.png
    fig_jn_cross_model_engagement.png
    fig_jn_cross_model_delta.png

Output dir: outputs/new_experiment/RQ2/
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
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import (INFERENTIAL_MODELS, IMPLICIT_CONDITIONS, MODEL_LABELS,
                    CONDITION_LABELS, MODEL_COLORS, FIX_RANGE, BIN_LABELS,
                    OUTPUT_DIR, out_dir)
from loaders import load_all
from RQ2_implicit_framing.jn_helpers import get_effect_curve

warnings.filterwarnings("ignore")

DV_TITLE = {"engagement_score": "Engagement (points)",
            "difficulty_delta":  "Difficulty $\\delta$ (units)"}


def load_rq1_degrees() -> dict:
    path = os.path.join(OUTPUT_DIR, "RQ1", "bic_polynomial_results.csv")
    rq1 = pd.read_csv(path)
    dv_map = {"Engagement Score": "engagement_score",
              "Difficulty Delta": "difficulty_delta"}
    deg_col = "ferguson_degree" if "ferguson_degree" in rq1.columns \
              else "selected_degree"
    return {(str(r["model"]).strip(),
             dv_map.get(str(r["dv"]).strip(), str(r["dv"]).strip())):
            int(r[deg_col]) for _, r in rq1.iterrows()}


def plot_binned_heatmap(dv: str, fname: str) -> None:
    path = os.path.join(out_dir("RQ2"), "binned_mean_diff.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found — run Output 3 first")
        return
    df = pd.read_csv(path)
    col = f"meandiff_{dv}"

    models = [MODEL_LABELS[k] for k in INFERENTIAL_MODELS
              if MODEL_LABELS[k] in df["model"].unique()]
    conds  = [c for c in IMPLICIT_CONDITIONS if c in df["condition"].unique()]

    fig, axes = plt.subplots(len(conds), 1, figsize=(14, 5 * len(conds)),
                             squeeze=False)
    for ri, cond in enumerate(conds):
        ax  = axes[ri][0]
        sub = df[df["condition"] == cond]
        mat = np.full((len(models), len(BIN_LABELS)), np.nan)
        for j, m in enumerate(models):
            for k, b in enumerate(BIN_LABELS):
                v = sub[(sub["model"] == m) & (sub["bin"] == b)][col]
                if len(v):
                    mat[j, k] = float(v.iloc[0])

        vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im   = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(BIN_LABELS)))
        ax.set_xticklabels(BIN_LABELS, rotation=45, fontsize=8)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(f"Mean difference — {CONDITION_LABELS.get(cond, cond)} "
                     f"vs Baseline ({DV_TITLE[dv]})",
                     fontsize=11, fontweight="bold")
        for j in range(len(models)):
            for k in range(len(BIN_LABELS)):
                if not np.isnan(mat[j, k]):
                    ax.text(k, j, f"{mat[j,k]:+.2f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, fraction=0.02, label=f"Δ {DV_TITLE[dv]}")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir("RQ2"), fname), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_jn_cross_model(dfs: dict, dv: str, fname: str) -> None:
    deg = load_rq1_degrees()
    conds = IMPLICIT_CONDITIONS
    fig, axes = plt.subplots(1, len(conds), figsize=(9 * len(conds), 6))
    if len(conds) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conds):
        ax.axhline(0, color="#555555", lw=0.9, ls="--", zorder=1)
        ax.set_xlim(0, 100)
        yall = []
        for model_key in INFERENTIAL_MODELS:
            if model_key not in dfs:
                continue
            label = MODEL_LABELS[model_key]
            color = MODEL_COLORS.get(label, "#888888")
            k = deg.get((label, dv), 1)
            res = get_effect_curve(dfs[model_key], cond, "baseline",
                                   dv=dv, degree=k)
            if res is None:
                continue
            eff, se, jn = res
            ax.plot(FIX_RANGE, eff, color=color, lw=2, label=label, zorder=3)
            ax.fill_between(FIX_RANGE, eff - 1.96 * se, eff + 1.96 * se,
                            color=color, alpha=0.10, zorder=2)
            for jp in jn["jn_points"]:
                ax.axvline(jp, color=color, lw=0.8, ls=":", alpha=0.6, zorder=4)
            yall.extend([eff.min(), eff.max()])
        if yall:
            s = max(abs(min(yall)), abs(max(yall)), 0.1)
            ax.set_ylim(-s * 1.5, s * 1.5)
        ax.set_title(CONDITION_LABELS.get(cond, cond).replace("\n", " "),
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Fixation rate (%)", fontsize=9)
        ax.set_ylabel(f"Simple effect ({dv})", fontsize=9)

    handles = [plt.Line2D([0], [0],
               color=MODEL_COLORS.get(MODEL_LABELS[k], "#888"), lw=2,
               label=MODEL_LABELS[k])
               for k in INFERENTIAL_MODELS if k in dfs]
    handles += [plt.Line2D([0], [0], color="grey", lw=0.8, ls=":",
                label="JN boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"JN Cross-Model: Implicit Framing — {dv}", fontsize=13,
                 fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(os.path.join(out_dir("RQ2"), fname), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    print("RQ2 | Output 4: Figures\n" + "=" * 60)
    plot_binned_heatmap("engagement_score", "fig_meandiff_binned_engagement.png")
    plot_binned_heatmap("difficulty_delta",  "fig_meandiff_binned_delta.png")
    dfs = load_all()
    plot_jn_cross_model(dfs, "engagement_score",
                        "fig_jn_cross_model_engagement.png")
    plot_jn_cross_model(dfs, "difficulty_delta",
                        "fig_jn_cross_model_delta.png")