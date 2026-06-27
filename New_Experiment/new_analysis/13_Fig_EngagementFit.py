"""
13_Fig_EngagementFit.py

Baseline engagement vs. fixation rate.

Output: outputs/Figures/RQ1/fig_engagement_<model>.png
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from config import (DATA_DIR, MODEL_FILES, MODEL_LABELS, MODEL_COLORS,
                    ALL_MODELS, out_dir)

warnings.filterwarnings("ignore")

DV = "engagement_score"

FORMULAS = {
    1: "{dv} ~ fixation_rate",
    2: "{dv} ~ fixation_rate + I(fixation_rate**2)",
    3: "{dv} ~ fixation_rate + I(fixation_rate**2) + I(fixation_rate**3)",
}

SHAPE_NAMES = {1: "Linear", 2: "Quadratic", 3: "Sigmoidal"}
FERGUSON_THRESHOLD = 0.04   # minimum practical R^2 gain (Ferguson, 2009)
X_GRID = np.linspace(0, 100, 500)

# The original 7 inferential models (excludes Llama-1B and the 32B models).
CORE_MODELS = [k for k in ALL_MODELS
              if k not in ("Llama-3.2-1B", "Qwen2.5-32B", "Qwen2.5-VL-32B")]

# For overview grid ordering: family grouping (sort order) and size (for
# sorting within a family). QwenVL counts as Qwen family.
PARAMS_B = {
    "Llama-1B": 1, "Llama-3B": 3, "Llama-8B": 8, "Llama-11B": 11,
    "Mistral-7B": 7,
    "Qwen-7B": 7, "Qwen-14B": 14, "QwenVL-7B": 7, "Qwen-32B": 32, "QwenVL-32B": 32,
}


def _family(label: str) -> int:
    if label.startswith("Llama"):
        return 0
    if label.startswith("Mistral"):
        return 2
    if label.startswith("Qwen"):
        return 1
    return 3


def _grouped_with_padding(model_keys: list[str], ncols: int) -> list[str | None]:
    """Group by family (Llama, Mistral, Qwen), sorted by parameter size
    within each family, with each family's slot count padded up to a
    multiple of ncols so families never share a row -- Mistral (a family of
    one) ends up alone on its own row rather than sharing with a neighbour."""
    by_family: dict[int, list[str]] = {}
    for k in model_keys:
        label = MODEL_LABELS.get(k, k)
        by_family.setdefault(_family(label), []).append(k)

    ordered: list[str | None] = []
    for fam in sorted(by_family):
        group = sorted(by_family[fam],
                       key=lambda k: PARAMS_B.get(MODEL_LABELS.get(k, k), 0))
        ordered.extend(group)
        ordered.extend([None] * ((-len(group)) % ncols))
    return ordered


def _plot_degree_curves(ax, fits: dict, sel: int, color: str,
                        lw_sel: float, lw_rej: float, label: bool) -> None:
    """Plot every degree's curve in FIXED order (1, 2, 3 -> Linear, Quadratic,
    Sigmoidal), regardless of which one is selected, so the legend/plot order
    never depends on the selection result."""
    for deg in sorted(fits):
        if deg == sel:
            ax.plot(X_GRID, fits[deg]["curve"], lw=lw_sel, color=color, zorder=4,
                    label=f"{SHAPE_NAMES[deg]} (selected)" if label else None)
        else:
            ax.plot(X_GRID, fits[deg]["curve"], lw=lw_rej, color="#bbbbbb",
                    ls="--", zorder=3,
                    label=f"{SHAPE_NAMES[deg]} (rejected)" if label else None)


def _load_baseline(model_key: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[df["parse_error"].isna() | (df["parse_error"].astype(str).isin(["", "nan", "None"]))]
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["engagement_score", "difficulty_delta", "fixation_rate", "condition"])
    df = df[(df.engagement_score >= 1) & (df.engagement_score <= 10)
            & (df.difficulty_delta >= -1) & (df.difficulty_delta <= 1)]
    return df[df["condition"] == "baseline"]


def fit_all_degrees(baseline: pd.DataFrame) -> dict:
    fits = {}
    for deg, formula in FORMULAS.items():
        mod = smf.ols(formula.format(dv=DV), data=baseline).fit()
        curve = mod.predict(pd.DataFrame({"fixation_rate": X_GRID})).values
        fits[deg] = {"model": mod, "r2": round(mod.rsquared, 4), "curve": curve}
    return fits


def ferguson_degree(fits: dict, threshold: float = FERGUSON_THRESHOLD) -> int:
    """Two-condition practical screen (identical to 04_FergusonSelection.py)."""
    base = None
    for deg in range(2, max(fits) + 1):
        if fits[deg]["r2"] - fits[1]["r2"] >= threshold:
            base = deg
            break
    if base is None:
        return 1

    accepted = base
    for deg in range(base + 1, max(fits) + 1):
        if fits[deg]["r2"] - fits[accepted]["r2"] >= threshold:
            accepted = deg
        else:
            break
    return accepted


def plot_model(model_key: str) -> None:
    baseline = _load_baseline(model_key)
    label = MODEL_LABELS.get(model_key, model_key)
    color = MODEL_COLORS.get(label, "#1f77b4")

    fits = fit_all_degrees(baseline)
    sel  = ferguson_degree(fits)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    # raw observations (n ~1010), not per-fixation-rate cell means
    ax.scatter(baseline["fixation_rate"], baseline[DV], s=8, alpha=0.15,
              color=color, zorder=2, linewidths=0)

    _plot_degree_curves(ax, fits, sel, color, lw_sel=2.4, lw_rej=1.2, label=True)

    ax.set_title(f"Fixation vs. Engagement (Baseline Condition) - {label}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Fixation Rate (%)", fontsize=9)
    ax.set_ylabel("Engagement Score", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=7.5, loc="best")

    plt.tight_layout()
    fname = f"fig_engagement_{label.replace(' ', '_')}.png"
    path = os.path.join(out_dir("Figures", "RQ1"), fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {label:<12} d{sel} {SHAPE_NAMES[sel]:<10} -> {fname}")


def plot_overview(model_keys: list[str], fname: str, suptitle_suffix: str) -> None:
    """One consolidated figure: every model's panel (scatter + rejected grey
    + selected colour) as a subplot in a single grid image. Models are
    grouped by family (Llama, Mistral, Qwen) and sorted by size within each
    family, with Mistral padded onto its own row since it's a family of one."""
    ncols = 3
    ordered = _grouped_with_padding(model_keys, ncols)
    nrows = len(ordered) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows),
                             sharex=True)
    axes_flat = axes.flatten()

    for i, model_key in enumerate(ordered):
        ax = axes_flat[i]
        if model_key is None:
            ax.set_visible(False)
            continue

        baseline = _load_baseline(model_key)
        label = MODEL_LABELS.get(model_key, model_key)
        color = MODEL_COLORS.get(label, "#1f77b4")

        fits = fit_all_degrees(baseline)
        sel  = ferguson_degree(fits)

        ax.scatter(baseline["fixation_rate"], baseline[DV], s=6, alpha=0.12,
                  color=color, zorder=2, linewidths=0)
        _plot_degree_curves(ax, fits, sel, color, lw_sel=2.0, lw_rej=1.0, label=True)

        ax.set_title(f"{label} ({SHAPE_NAMES[sel]})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Fixation Rate (%)", fontsize=8.5)
        ax.set_ylabel("Engagement Score", fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8.5, loc="lower right")

    fig.suptitle(f"Fixation vs. Engagement (Baseline Condition)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir("Figures", "RQ1"), fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved consolidated overview -> {fname}")


if __name__ == "__main__":
    print("13 | Baseline Engagement Fit (one figure per model)\n" + "=" * 60)
    available = []
    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            print(f"  [MISSING] {MODEL_FILES[model_key]}")
            continue
        plot_model(model_key)
        available.append(model_key)

    plot_overview(available, "fig_engagement_overview_all.png", "All Models")

    core_available = [k for k in available if k in CORE_MODELS]
    plot_overview(core_available, "fig_engagement_overview_core.png", "Core Models")

    print(f"\nSaved figures to {out_dir('Figures', 'RQ1')}")
