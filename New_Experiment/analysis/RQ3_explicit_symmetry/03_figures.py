"""
RQ3 | Output 3: Explicit framing figures.

Polynomial JN at the RQ1 Ferguson-retained degree per (model, DV), matching
the RQ2 figures and the RQ3 omnibus. Produces:

    fig_explicit_comparison.png          baseline vs GA vs DGE, all models
    fig_jn_ga_cross_model_{dv}.png       JN curves, GA vs baseline, per DV
    fig_jn_dge_cross_model_{dv}.png      JN curves, DGE vs baseline, per DV
    fig_jn_ga_{model}.png                per-model JN, GA vs baseline (both DVs)

The symmetry-profile figure (A = d_DGE + d_GA) is retired: symmetry is
answered by the GA-vs-DGE eta^2 contrast in the omnibus.

Output dir: outputs/new_experiment/RQ3/
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import (INFERENTIAL_MODELS, MODEL_LABELS, MODEL_COLORS,
                    CONDITION_LABELS, COL, FIX_RANGE, OUTPUT_DIR, out_dir)
from loaders import load_all, cell_means

sys.path.insert(0, os.path.dirname(__file__))
from RQ3jn_helpers import get_effect_curve

warnings.filterwarnings("ignore")

DVS = [("engagement_score", "Engagement Score"),
       ("difficulty_delta",  "Difficulty Delta")]

EXPLICIT_CUES = [
    ("gaze_aversion",        COL["gaze_aversion"],        "ga",  "Gaze Aversion"),
    ("direct_gaze_explicit", COL["direct_gaze_explicit"], "dge", "Direct Gaze"),
]


def load_rq1_degrees() -> dict:
    path = os.path.join(OUTPUT_DIR, "RQ1", "bic_polynomial_results.csv")
    rq1 = pd.read_csv(path)
    dv_map = {"Engagement Score": "engagement_score",
              "Difficulty Delta": "difficulty_delta"}
    deg_col = ("ferguson_degree" if "ferguson_degree" in rq1.columns
               else "selected_degree")
    return {(str(r["model"]).strip(),
             dv_map.get(str(r["dv"]).strip(), str(r["dv"]).strip())):
            int(r[deg_col]) for _, r in rq1.iterrows()}


def plot_explicit_comparison(dfs: dict) -> None:
    """2 rows (DVs) x 2 cols (cues): all models overlaid, baseline dashed grey."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Explicit Framing vs Baseline — all models overlaid",
                 fontsize=12, fontweight="bold")

    for row_i, (dv, dv_label) in enumerate(DVS):
        for col_i, (cond, cond_color, _, cue_label) in enumerate(EXPLICIT_CUES):
            ax = axes[row_i][col_i]
            for model_key in INFERENTIAL_MODELS:
                if model_key not in dfs:
                    continue
                label = MODEL_LABELS[model_key]
                color = MODEL_COLORS.get(label, "#888888")
                df_m  = dfs[model_key]
                for condition, ls, lw, c in [
                    ("baseline", "--", 0.9, "#aaaaaa"),
                    (cond,       "-",  1.8, color),
                ]:
                    sub = df_m[df_m["condition"] == condition]
                    if len(sub) == 0:
                        continue
                    cm = cell_means(sub, dv)
                    ax.plot(cm["fixation_rate"], cm[f"{dv}_mean"],
                            color=c, lw=lw, ls=ls,
                            alpha=0.9 if condition != "baseline" else 0.4,
                            label=label if condition != "baseline" else None)
            ax.axhline(0, color="#999999", lw=0.7, ls=":")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Fixation rate (%)", fontsize=9)
            ax.set_ylabel(dv_label, fontsize=9)
            ax.set_title(f"{cue_label} vs Baseline", fontsize=10, fontweight="bold")

    handles = [plt.Line2D([0], [0], color=MODEL_COLORS.get(MODEL_LABELS[k], "#888"),
                          lw=2, label=MODEL_LABELS[k])
               for k in INFERENTIAL_MODELS if k in dfs]
    handles += [plt.Line2D([0], [0], color="#aaaaaa", lw=0.9, ls="--",
                           label="Baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8,
               frameon=True, bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(os.path.join(out_dir("RQ3"), "fig_explicit_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_explicit_comparison.png")


def plot_jn_cross_model(dfs: dict, cond: str, cue_tag: str,
                        cue_label: str, dv: str, dv_label: str,
                        deg: dict) -> None:
    """All-models JN overlay for one cue, one DV, at the Ferguson degree."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        label = MODEL_LABELS[model_key]
        color = MODEL_COLORS.get(label, "#888888")
        k = deg.get((label, dv), 1)
        res = get_effect_curve(dfs[model_key], cond, "baseline", dv=dv, degree=k)
        if res is None:
            continue
        effect, se, jn = res
        ax.plot(FIX_RANGE, effect, color=color, lw=1.8, alpha=0.85, label=label)
        ax.fill_between(FIX_RANGE, effect - 1.96 * se, effect + 1.96 * se,
                        color=color, alpha=0.10)
        for jp in jn["jn_points"]:
            ax.axvline(jp, color=color, lw=0.9, ls=":", alpha=0.6)
    ax.axhline(0, color=COL["zero"], lw=1.0, ls="--")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fixation rate (%)", fontsize=9)
    ax.set_ylabel(f"Simple effect on {dv_label}", fontsize=9)
    ax.set_title(f"JN: {cue_label} vs Baseline — {dv_label}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()
    fname = f"fig_jn_{cue_tag}_cross_model_{dv}.png"
    fig.savefig(os.path.join(out_dir("RQ3"), fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_jn_per_model(dfs: dict, cond: str, cue_tag: str,
                      cue_label: str, deg: dict) -> None:
    """Per-model JN for one cue, both DVs side by side, at the Ferguson degree."""
    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        label = MODEL_LABELS[model_key]
        df_m  = dfs[model_key]
        fig, axes = plt.subplots(1, len(DVS), figsize=(14, 5))
        fig.suptitle(f"{label} — {cue_label} vs Baseline",
                     fontsize=11, fontweight="bold")
        for ax, (dv, dv_label) in zip(axes, DVS):
            k = deg.get((label, dv), 1)
            res = get_effect_curve(df_m, cond, "baseline", dv=dv, degree=k)
            if res is None:
                ax.set_visible(False)
                continue
            effect, se, jn = res
            mask = jn["sig_mask"]
            ax.fill_between(FIX_RANGE, effect - 1.96 * se, effect + 1.96 * se,
                            where=mask, color=COL["sig"], alpha=0.45,
                            label="Significant")
            ax.fill_between(FIX_RANGE, effect - 1.96 * se, effect + 1.96 * se,
                            color=COL["ci"], alpha=0.25, label="95% CI")
            ax.plot(FIX_RANGE, effect, color=COL[cond], lw=2.0,
                    label=f"{cue_label} − Baseline")
            ax.axhline(0, color=COL["zero"], lw=0.9, ls="--")
            for jp in jn["jn_points"]:
                ax.axvline(jp, color=COL["jn"], lw=1.2, ls=":")
            ax.set_xlabel("Fixation rate (%)", fontsize=9)
            ax.set_ylabel(f"Effect on {dv_label}", fontsize=9)
            ax.set_title(f"{dv_label}  (k={k})", fontsize=9)
            ax.set_xlim(0, 100)
            ax.legend(fontsize=7, loc="best")
        plt.tight_layout()
        safe = model_key.replace("/", "_").replace(".", "_")
        fname = f"fig_jn_{cue_tag}_{safe}.png"
        fig.savefig(os.path.join(out_dir("RQ3"), fname), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


if __name__ == "__main__":
    print("RQ3 | Output 3: Figures\n" + "=" * 60)
    dfs = load_all()
    deg = load_rq1_degrees()

    plot_explicit_comparison(dfs)

    for cond, _color, tag, cue_label in EXPLICIT_CUES:
        for dv, dv_label in DVS:
            plot_jn_cross_model(dfs, cond, tag, cue_label, dv, dv_label, deg)

    
    #plot_jn_per_model(dfs, "gaze_aversion", "ga", "Gaze Aversion", deg)
    
    plot_jn_per_model(dfs, "gaze_aversion", "ga",  "Gaze Aversion", deg)
    plot_jn_per_model(dfs, "direct_gaze_explicit", "dge", "Direct Gaze",   deg)