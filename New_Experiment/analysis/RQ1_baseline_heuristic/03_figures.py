"""
RQ1 | Output 3: Baseline fixation curves — engagement and difficulty delta.

Curve selection uses the same two-condition Ferguson screen as the results
table (Output 1), fitted on the raw baseline observations, so the figures and
the tables can never disagree on degree.

Engagement: selected curve only (the result is "six linear, one quadratic";
showing rejected higher-order fits would imply they are contenders).
Difficulty: all three degrees drawn, rejected ones greyed, selected one in
colour, so the term-by-term retention of the two cubic sigmoidals is visible.

Produces:
    fig_baseline_engagement.png
    fig_baseline_delta.png
    fig_baseline_summary.png

Output dir: outputs/new_experiment/RQ1/
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
                    FIX_RANGE, out_dir)
from loaders import load_all, cell_means

warnings.filterwarnings("ignore")

FERGUSON_THRESHOLD = 0.04


def fit_degrees(x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> dict:
    """Fit degrees 1..max_degree on the given data; return coeffs, r2, and
    the fitted curve over FIX_RANGE for each degree."""
    fits = {}
    for deg in range(1, max_degree + 1):
        coeffs = np.polyfit(x, y, deg)
        sse    = float(np.sum((y - np.polyval(coeffs, x)) ** 2))
        sst    = float(np.sum((y - y.mean()) ** 2))
        r2     = 1.0 - sse / sst if sst > 0 else 0.0
        fits[deg] = {"coeffs": coeffs, "r2": r2,
                     "curve": np.polyval(coeffs, FIX_RANGE)}
    return fits


def ferguson_degree(fits: dict, threshold: float = FERGUSON_THRESHOLD) -> int:
    """Two-condition screen, identical to Output 1: lowest degree gaining
    >= threshold over linear, then extend only while each further term gains
    >= threshold over the current accepted degree."""
    base = None
    for deg in range(2, max(fits) + 1):
        if fits[deg]["r2"] - fits[1]["r2"] >= threshold:
            base = deg
            break
    if base is None:
        return 1
    sel = base
    for deg in range(base + 1, max(fits) + 1):
        if fits[deg]["r2"] - fits[sel]["r2"] >= threshold:
            sel = deg
        else:
            break
    return sel


def get_fits_and_selection(dfs: dict, model_key: str, dv: str):
    """Select degree on RAW baseline data (matching the table); return the
    full fit dict, the selected degree, and the cell means for scatter."""
    bl = dfs[model_key][dfs[model_key]["condition"] == "baseline"]
    x_raw = bl["fixation_rate"].values.astype(float)
    y_raw = bl[dv].values.astype(float)
    fits  = fit_degrees(x_raw, y_raw)
    sel   = ferguson_degree(fits)
    cm    = cell_means(bl, dv)        # for scatter only
    return fits, sel, cm


def plot_per_dv(dfs: dict, dv: str, ylabel: str, fname: str,
                show_rejected: bool) -> None:
    models = [m for m in INFERENTIAL_MODELS if m in dfs]
    ncols  = 4
    nrows  = -(-len(models) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows),
                             sharey=False, sharex=True)
    axes_flat = axes.flatten()

    shape_names = {1: "Linear", 2: "Quadratic", 3: "Cubic"}

    for i, model_key in enumerate(models):
        ax    = axes_flat[i]
        label = MODEL_LABELS[model_key]
        color = MODEL_COLORS.get(label, "#1f77b4")
        fits, sel, cm = get_fits_and_selection(dfs, model_key, dv)

        x_scatter = cm["fixation_rate"].values.astype(float)
        y_scatter = cm[f"{dv}_mean"].values

        ax.scatter(x_scatter, y_scatter, s=8, alpha=0.5,
                   color=color, zorder=2)

        # optionally draw the rejected degrees in grey
        if show_rejected:
            for deg in fits:
                if deg == sel:
                    continue
                ax.plot(FIX_RANGE, fits[deg]["curve"], lw=1.2,
                        color="#bbbbbb", ls="--", zorder=3,
                        label=f"{shape_names[deg]} (rejected)")

        # selected curve, solid, in colour
        ax.plot(FIX_RANGE, fits[sel]["curve"], lw=2.4, color=color, zorder=4,
                label=f"{shape_names[sel]} (selected)")

        ax.axhline(0, color="#aaaaaa", lw=0.7, ls="--", zorder=1)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Fixation rate (%)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc="upper left")

    for j in range(len(models), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Baseline Condition: Fixation Rate vs {ylabel}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir("RQ1"), fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_summary(dfs: dict) -> None:
    """All models overlaid (selected curve only), both DVs side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for dv, ax, ylabel in [
        ("engagement_score", ax1, "Engagement Score"),
        ("difficulty_delta",  ax2, "Difficulty Delta"),
    ]:
        for model_key in INFERENTIAL_MODELS:
            if model_key not in dfs:
                continue
            label = MODEL_LABELS[model_key]
            fits, sel, _ = get_fits_and_selection(dfs, model_key, dv)
            ax.plot(FIX_RANGE, fits[sel]["curve"], lw=2,
                    color=MODEL_COLORS.get(label, "#888888"), label=label)

        ax.axhline(0, color="#aaaaaa", lw=0.8, ls="--")
        ax.set_xlabel("Fixation rate (%)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Baseline: Fixation vs {ylabel}", fontsize=11)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 100)

    plt.tight_layout()
    path = os.path.join(out_dir("RQ1"), "fig_baseline_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_baseline_summary.png")


if __name__ == "__main__":
    print("RQ1 | Output 3: Baseline Figures\n" + "=" * 60)
    dfs = load_all()
    # engagement: selected curve only
    plot_per_dv(dfs, "engagement_score", "Engagement Score",
                "fig_baseline_engagement.png", show_rejected=False)
    # difficulty: show rejected degrees greyed, selected highlighted
    plot_per_dv(dfs, "difficulty_delta", "Difficulty Delta",
                "fig_baseline_delta.png", show_rejected=True)
    plot_summary(dfs)