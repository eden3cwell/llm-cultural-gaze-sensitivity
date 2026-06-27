"""
18_Binned_Means.py

Binned mean-difference heatmaps for both the implicit and explicit framing
conditions.

Output:
    outputs/Figures/RQ2/fig_meandiff_binned_<dv>.png   (implicit)
    outputs/Figures/RQ3/fig_meandiff_binned_<dv>.png   (explicit)
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(__file__))
from config import (ALL_MODELS, MODEL_LABELS, CONDITION_LABELS, BIN_LABELS,
                    IMPLICIT_CONDITIONS, EXPLICIT_CONDITIONS, OUTPUT_DIR, out_dir)

warnings.filterwarnings("ignore")

DVS = {"engagement_score": "Engagement (points)", "difficulty_delta": "Difficulty Delta (units)"}

FRAMINGS = [
    ("Implicit", "RQ2", IMPLICIT_CONDITIONS),
    ("Explicit", "RQ3", EXPLICIT_CONDITIONS),
]


def plot_binned_heatmap(csv_path: str, conditions: list[str], dv: str, dv_label: str,
                        out_subdir: str, fname: str) -> None:
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {csv_path} not found")
        return
    df = pd.read_csv(csv_path)
    col = f"meandiff_{dv}"

    all_labels = [MODEL_LABELS.get(k, k) for k in ALL_MODELS]
    models = [m for m in all_labels if m in df["model"].unique()]
    conds  = [c for c in conditions if c in df["condition"].unique()]
    if not models or not conds:
        print(f"  [SKIP] {fname}: no matching models/conditions")
        return

    fig, axes = plt.subplots(len(conds), 1, figsize=(14, 5 * len(conds)), squeeze=False)
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
        ax.set_title(f"Mean difference -- {CONDITION_LABELS.get(cond, cond)} "
                     f"vs Baseline ({dv_label})",
                     fontsize=11, fontweight="bold")
        for j in range(len(models)):
            for kk in range(len(BIN_LABELS)):
                if not np.isnan(mat[j, kk]):
                    ax.text(kk, j, f"{mat[j, kk]:+.2f}", ha="center", va="center",
                            fontsize=7, color="black")
        plt.colorbar(im, ax=ax, fraction=0.02, label=f"Delta {dv_label}")

    plt.tight_layout()
    path = os.path.join(out_dir("Figures", out_subdir), fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    print("18 | Binned Mean-Difference Heatmaps\n" + "=" * 60)
    for framing_label, rq, conditions in FRAMINGS:
        csv_path = os.path.join(OUTPUT_DIR, rq, "binned_mean_diff.csv")
        for dv, dv_label in DVS.items():
            plot_binned_heatmap(csv_path, conditions, dv, dv_label, rq,
                               f"fig_meandiff_binned_{dv}.png")

    print(f"\nSaved figures to {out_dir('Figures', 'RQ2')} and {out_dir('Figures', 'RQ3')}")
