"""
Robustness Check | Output 2: Prompt robustness figures.

Overlays three polynomial curves (one per prompt variant) for the
baseline condition of each representative model.

Variant sources:
    standard   — main model CSV (baseline rows)
    variant_a  — robustness_check.csv
    variant_b  — robustness_check.csv

Outputs:
    outputs/new_experiment/robustness/fig_robustness_Llama-3B.png
    outputs/new_experiment/robustness/fig_robustness_Qwen-14B.png
    outputs/new_experiment/robustness/fig_robustness_overlay.png
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
from config import MODEL_LABELS, out_dir
from loaders import load_model, load_robustness

warnings.filterwarnings("ignore")

ROBUSTNESS_MODEL_MAP = {
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
}

VARIANTS = ["standard", "variant_a", "variant_b"]
DVS = [("engagement_score", "Engagement Score"),
       ("difficulty_delta",  "Difficulty Delta")]

VARIANT_STYLES = {
    "standard":  {"color": "#333333", "lw": 2.2, "ls": "-",  "label": "Standard"},
    "variant_a": {"color": "#e6812e", "lw": 1.8, "ls": "--", "label": "Variant A"},
    "variant_b": {"color": "#3a7abf", "lw": 1.8, "ls": ":",  "label": "Variant B"},
}

X_RANGE       = np.linspace(0, 100, 500)
SCATTER_SIZE  = 12
SCATTER_ALPHA = 0.25


def bic_degree(x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> int:
    n = len(x)
    best_bic, best_deg = np.inf, 1
    for deg in range(1, max_degree + 1):
        c   = np.polyfit(x, y, deg)
        sse = float(np.sum((y - np.polyval(c, x)) ** 2))
        bic = n * np.log(sse / n) + (deg + 1) * np.log(n)
        if bic < best_bic:
            best_bic, best_deg = bic, deg
    return best_deg


def load_model_variants(
    model_key: str,
    df_rob: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    full_id = ROBUSTNESS_MODEL_MAP[model_key]
    result  = {}

    df_main = load_model(model_key)
    if df_main is not None:
        std = df_main[df_main["condition"] == "baseline"].copy()
        if len(std) > 0:
            result["standard"] = std

    if df_rob is not None and len(df_rob) > 0:
        for variant in ("variant_a", "variant_b"):
            sub = df_rob[
                (df_rob["model"] == full_id) &
                (df_rob["prompt_variant"] == variant)
            ].copy()
            if len(sub) > 0:
                result[variant] = sub

    return result


def draw_variant_panel(
    ax: plt.Axes,
    variants: dict[str, pd.DataFrame],
    dv: str,
    dv_label: str,
    model_label: str,
) -> None:
    for variant in VARIANTS:
        sub = variants.get(variant)
        if sub is None or len(sub) == 0:
            continue

        style = VARIANT_STYLES[variant]
        cm    = sub.groupby("fixation_rate")[dv].mean().reset_index()
        x_r   = cm["fixation_rate"].values.astype(float)
        y_r   = cm[dv].values

        ax.scatter(x_r, y_r,
                   color=style["color"], s=SCATTER_SIZE,
                   alpha=SCATTER_ALPHA, zorder=2)

        deg    = bic_degree(x_r, y_r)
        coeffs = np.polyfit(x_r, y_r, deg)
        y_fit  = np.polyval(coeffs, X_RANGE)

        ax.plot(X_RANGE, y_fit,
                color=style["color"],
                lw=style["lw"],
                ls=style["ls"],
                label=f"{style['label']} (deg {deg})",
                zorder=3)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Fixation rate (%)", fontsize=9)
    ax.set_ylabel(dv_label, fontsize=9)
    ax.set_title(f"{model_label} — {dv_label}", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, loc="best")


def plot_per_model(
    model_key: str,
    variants: dict[str, pd.DataFrame],
) -> None:
    label = MODEL_LABELS.get(model_key, model_key)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Prompt Robustness — {label}\n"
        "Baseline: three prompt variants overlaid",
        fontsize=11, fontweight="bold"
    )

    for ax, (dv, dv_label) in zip(axes, DVS):
        draw_variant_panel(ax, variants, dv, dv_label, label)

    plt.tight_layout()
    safe = label.replace("/", "_").replace(".", "_").replace(" ", "")
    path = os.path.join(out_dir("robustness"), f"fig_robustness_{safe}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_robustness_{safe}.png")


def plot_overlay(all_variants: dict[str, dict[str, pd.DataFrame]]) -> None:
    keys  = list(all_variants.keys())
    fig, axes = plt.subplots(len(keys), 2, figsize=(14, 5 * len(keys)))
    if len(keys) == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "Prompt Robustness — Both Representative Models\n"
        "Solid = standard  |  Dashed = variant A  |  Dotted = variant B",
        fontsize=11, fontweight="bold"
    )

    for row_i, model_key in enumerate(keys):
        label    = MODEL_LABELS.get(model_key, model_key)
        variants = all_variants[model_key]
        for col_i, (dv, dv_label) in enumerate(DVS):
            draw_variant_panel(axes[row_i][col_i], variants, dv, dv_label, label)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir("robustness"), "fig_robustness_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_robustness_overlay.png")


if __name__ == "__main__":
    print("Robustness Check | Output 2: Figures\n" + "=" * 60)

    df_rob = load_robustness()
    if df_rob is None:
        print("  [WARN] robustness_check.csv not found — "
              "only 'standard' variant will be plotted if main CSVs exist.")
        df_rob = pd.DataFrame()

    all_variants = {}
    for model_key in ROBUSTNESS_MODEL_MAP:
        v = load_model_variants(model_key, df_rob)
        if v:
            all_variants[model_key] = v
            plot_per_model(model_key, v)
        else:
            print(f"  [SKIP] {MODEL_LABELS.get(model_key, model_key)}: no data")

    if all_variants:
        plot_overlay(all_variants)
    else:
        print("  [ERROR] No data found for any model — "
              "check that main model CSVs exist.")
