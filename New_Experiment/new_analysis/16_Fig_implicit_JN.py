"""
16_Fig_implicit_JN.py

Polynomial Johnson-Neyman (floodlight) analysis for the implicit framing
conditions (maori, kaumatua).

Output:
    outputs/Figures/RQ2/fig_jn_<model>.png
    outputs/Figures/RQ2/fig_jn_cross_model_<dv>_all.png
    outputs/Figures/RQ2/fig_jn_cross_model_<dv>_core.png
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, os.path.dirname(__file__))
from config import (DATA_DIR, MODEL_FILES, MODEL_LABELS, MODEL_COLORS,
                    ALL_MODELS, CONDITION_LABELS, COL, ALPHA, out_dir)

warnings.filterwarnings("ignore")

DVS = {"engagement_score": "Engagement", "difficulty_delta": "Difficulty Delta"}

FORMULAS = {
    1: "{dv} ~ fixation_rate",
    2: "{dv} ~ fixation_rate + I(fixation_rate**2)",
    3: "{dv} ~ fixation_rate + I(fixation_rate**2) + I(fixation_rate**3)",
}

FERGUSON_THRESHOLD = 0.04
X_GRID = np.linspace(0, 100, 500)

JN_PAIRS = [
    ("maori",    "baseline"),
    ("kaumatua", "baseline"),
    ("kaumatua", "maori"),
]

# The original 7 inferential models (excludes Llama-1B and the 32B models).
CORE_MODELS = [k for k in ALL_MODELS
              if k not in ("Llama-3.2-1B", "Qwen2.5-32B", "Qwen2.5-VL-32B")]


def _load_clean(model_key: str) -> pd.DataFrame:
    """All conditions, cleaned."""
    path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[df["parse_error"].isna() | (df["parse_error"].astype(str).isin(["", "nan", "None"]))]
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["engagement_score", "difficulty_delta", "fixation_rate", "condition"])
    df = df[(df.engagement_score >= 1) & (df.engagement_score <= 10)
            & (df.difficulty_delta >= -1) & (df.difficulty_delta <= 1)]
    return df


def ferguson_degree_for(dv: str, baseline: pd.DataFrame) -> int:
    """Two-condition practical screen (identical to 04_FergusonSelection.py),
    fit on baseline only."""
    fits = {}
    for deg, formula in FORMULAS.items():
        mod = smf.ols(formula.format(dv=dv), data=baseline).fit()
        fits[deg] = {"r2": round(mod.rsquared, 4)}

    base = None
    for deg in range(2, max(fits) + 1):
        if fits[deg]["r2"] - fits[1]["r2"] >= FERGUSON_THRESHOLD:
            base = deg
            break
    if base is None:
        return 1

    accepted = base
    for deg in range(base + 1, max(fits) + 1):
        if fits[deg]["r2"] - fits[accepted]["r2"] >= FERGUSON_THRESHOLD:
            accepted = deg
        else:
            break
    return accepted


def get_effect_curve(df: pd.DataFrame, treatment_cond: str, reference_cond: str,
                     dv: str, degree: int, alpha: float = ALPHA):
    """Fit DV ~ cond_bin * poly(fixation, degree) and return (effect, se, jn)
    over X_GRID. effect[i] is the simple effect (treatment - reference) at
    grid point i; jn carries the significance mask and JN boundaries."""
    subset = df[df["condition"].isin([reference_cond, treatment_cond])].copy()
    subset["cond_bin"] = (subset["condition"] == treatment_cond).astype(int)
    if subset["cond_bin"].nunique() < 2:
        return None

    poly_terms = " + ".join(
        ["fixation_rate"] + [f"I(fixation_rate**{j})" for j in range(2, degree + 1)]
    )
    formula = f"{dv} ~ cond_bin * ({poly_terms})"
    mod = smf.ols(formula, data=subset).fit()

    params, vcov = mod.params, mod.cov_params()
    t_crit = stats.t.ppf(1 - alpha / 2, int(mod.df_resid))

    eff_names = ["cond_bin", "cond_bin:fixation_rate"]
    eff_names += [f"cond_bin:I(fixation_rate ** {j})" for j in range(2, degree + 1)]
    eff_names = [n for n in eff_names if n in params.index]

    C = np.ones((len(X_GRID), len(eff_names)))
    for col, name in enumerate(eff_names):
        if name == "cond_bin":
            C[:, col] = 1.0
        elif name == "cond_bin:fixation_rate":
            C[:, col] = X_GRID
        else:
            j = int(name.split("**")[1].split(")")[0])
            C[:, col] = X_GRID ** j

    beta = params[eff_names].to_numpy()
    Sig  = vcov.loc[eff_names, eff_names].to_numpy()

    effect = C @ beta
    var    = np.maximum(np.einsum("ij,jk,ik->i", C, Sig, C), 1e-15)
    se     = np.sqrt(var)
    t_vals = effect / se

    sig_mask = np.abs(t_vals) >= t_crit
    flips = np.where(np.diff(sig_mask.astype(int)) != 0)[0]
    jn_points = sorted(float(X_GRID[i + 1]) for i in flips)

    jn = dict(sig_mask=sig_mask, jn_points=jn_points,
             always_sig=bool(sig_mask.all()), never_sig=bool(not sig_mask.any()))
    return effect, se, jn


def plot_model(model_key: str) -> None:
    """One PNG per model: DV (rows) x pair (cols) grid."""
    data = _load_clean(model_key)
    label = MODEL_LABELS.get(model_key, model_key)
    color = MODEL_COLORS.get(label, "#1f77b4")
    baseline = data[data["condition"] == "baseline"]

    fig, axes = plt.subplots(len(DVS), len(JN_PAIRS),
                             figsize=(5.2 * len(JN_PAIRS), 4.0 * len(DVS)))

    for r, (dv, dv_label) in enumerate(DVS.items()):
        k = ferguson_degree_for(dv, baseline)
        for c, (treat, ref) in enumerate(JN_PAIRS):
            ax = axes[r, c]
            res = get_effect_curve(data, treat, ref, dv, k)
            if res is None:
                ax.set_visible(False)
                continue
            eff, se, jn = res

            ax.fill_between(X_GRID, -1e3, 1e3, where=jn["sig_mask"],
                            color=COL["sig"], alpha=0.30, zorder=0)
            ax.fill_between(X_GRID, eff - 1.96 * se, eff + 1.96 * se,
                            color=COL["ci"], alpha=0.55, zorder=1)
            ax.plot(X_GRID, eff, color=color, lw=2, zorder=2)
            ax.axhline(0, color=COL["zero"], lw=0.9, ls="--", zorder=3)
            for jp in jn["jn_points"]:
                ax.axvline(jp, color=COL["jn"], lw=1.2, ls=":", zorder=4)

            yabs = max(abs(eff.max()), abs(eff.min()), 0.1)
            ax.set_ylim(-yabs * 1.6, yabs * 1.6)
            ax.set_xlim(0, 100)

            tl = CONDITION_LABELS.get(treat, treat).replace("\n", " ")
            rl = CONDITION_LABELS.get(ref, ref).replace("\n", " ")
            ax.set_title(f"{tl} vs {rl}", fontsize=9.5, fontweight="bold")
            ax.set_xlabel("Fixation Rate (%)", fontsize=8)
            # the degree k is shared across the whole row -- print it once,
            # on the leftmost panel only, instead of repeating it per pair.
            ylabel = f"Simple Effect ({dv_label})"
            if c == 0:
                ylabel += f"\n(k={k})"
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7.5)

    fig.suptitle(f"Johnson-Neyman: {label} (Implicit Framing)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fname = f"fig_jn_{label.replace(' ', '_')}.png"
    path = os.path.join(out_dir("Figures", "RQ2"), fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {label:<12} -> {fname}")


def plot_cross_model(dv: str, dv_label: str, model_keys: list[str],
                     fname: str, suptitle_suffix: str,
                     pairs: list[tuple[str, str]] = JN_PAIRS) -> None:
    """One PNG per DV: a single row of panels (one per pair), each overlaying
    every model's effect curve on the same axes."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.5 * len(pairs), 5.5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (treat, ref) in zip(axes, pairs):
        ax.axhline(0, color=COL["zero"], lw=0.9, ls="--", zorder=1)
        ax.set_xlim(0, 100)
        yall = []
        for model_key in model_keys:
            path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
            if not os.path.exists(path):
                continue
            data = _load_clean(model_key)
            label = MODEL_LABELS.get(model_key, model_key)
            color = MODEL_COLORS.get(label, "#888888")
            baseline = data[data["condition"] == "baseline"]
            k = ferguson_degree_for(dv, baseline)

            res = get_effect_curve(data, treat, ref, dv, k)
            if res is None:
                continue
            eff, se, jn = res
            ax.plot(X_GRID, eff, color=color, lw=2.2, alpha=0.75, zorder=3)
            ax.fill_between(X_GRID, eff - 1.96 * se, eff + 1.96 * se,
                            color=color, alpha=0.10, zorder=2)
            for jp in jn["jn_points"]:
                ax.axvline(jp, color=color, lw=0.8, ls=":", alpha=0.6, zorder=4)
            yall.extend([float(eff.min()), float(eff.max())])

        if yall:
            s = max(abs(min(yall)), abs(max(yall)), 0.1)
            ax.set_ylim(-s * 1.5, s * 1.5)

        tl = CONDITION_LABELS.get(treat, treat).replace("\n", " ")
        rl = CONDITION_LABELS.get(ref, ref).replace("\n", " ")
        ax.set_title(f"{tl} vs {rl}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Fixation Rate (%)", fontsize=10)
        ax.set_ylabel(f"Simple Effect ({dv_label})", fontsize=10)
        ax.tick_params(labelsize=9)

    handles = [mlines.Line2D([0], [0], color=MODEL_COLORS.get(MODEL_LABELS.get(k, k), "#888"),
              lw=2.5, label=MODEL_LABELS.get(k, k))
              for k in model_keys if os.path.exists(os.path.join(DATA_DIR, MODEL_FILES[k]))]
    handles += [mlines.Line2D([0], [0], color="grey", lw=1.0, ls=":",
                label="JN boundary")]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
              fontsize=11, frameon=True, framealpha=0.95,
              bbox_to_anchor=(0.5, -0.08))

    fig.suptitle(f"JN Cross-Model: Implicit Framing - {dv_label} ({suptitle_suffix})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    path = os.path.join(out_dir("Figures", "RQ2"), fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved cross-model -> {fname}")


if __name__ == "__main__":
    print("16 | Johnson-Neyman: Implicit Framing\n" + "=" * 60)
    available = []
    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            print(f"  [MISSING] {MODEL_FILES[model_key]}")
            continue
        plot_model(model_key)
        available.append(model_key)

    core_available = [k for k in available if k in CORE_MODELS]
    core_pairs = [("maori", "baseline"), ("kaumatua", "baseline")]
    for dv, dv_label in DVS.items():
        plot_cross_model(dv, dv_label, available,
                         f"fig_jn_cross_model_{dv}_all.png", "All Models")
        plot_cross_model(dv, dv_label, core_available,
                         f"fig_jn_cross_model_{dv}_core.png", "Core Models",
                         pairs=core_pairs)

    print(f"\nSaved figures to {out_dir('Figures', 'RQ2')}")
