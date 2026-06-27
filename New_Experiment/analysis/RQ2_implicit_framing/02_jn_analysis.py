"""
RQ2 | Output 2: Polynomial Johnson-Neyman — implicit conditions, both DVs.

For each model x condition-pair x DV, fits the interaction at the RQ1
Ferguson-retained polynomial degree and identifies fixation-rate regions of
significance via the polynomial floodlight in jn_helpers.get_effect_curve.

Pairs: maori vs baseline, kaumatua vs baseline, kaumatua vs maori (contrast).
No multiple-comparison correction (effect-size-led; p reported for reference
only, inert under near-deterministic output).

Outputs:
    outputs/new_experiment/RQ2/jn_summary.csv
    outputs/new_experiment/RQ2/fig_jn_{model_key}.png
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
from config import (INFERENTIAL_MODELS, IMPLICIT_CONDITIONS, MODEL_LABELS,
                    CONDITION_LABELS, FIX_RANGE, COL, OUTPUT_DIR, out_dir)
from loaders import load_all
from RQ2_implicit_framing.jn_helpers import get_effect_curve

warnings.filterwarnings("ignore")

DVS = ["engagement_score", "difficulty_delta"]

JN_PAIRS = [
    ("maori",    "baseline"),
    ("kaumatua", "baseline"),
    ("kaumatua", "maori"),
]

FALLBACK_DEG = {
    ("Llama-3B",  "engagement_score"): 1, ("Llama-3B",  "difficulty_delta"): 2,
    ("Llama-8B",  "engagement_score"): 1, ("Llama-8B",  "difficulty_delta"): 2,
    ("Llama-11B", "engagement_score"): 1, ("Llama-11B", "difficulty_delta"): 3,
    ("Mistral-7B","engagement_score"): 1, ("Mistral-7B","difficulty_delta"): 2,
    ("Qwen-7B",   "engagement_score"): 1, ("Qwen-7B",   "difficulty_delta"): 3,
    ("Qwen-14B",  "engagement_score"): 2, ("Qwen-14B",  "difficulty_delta"): 2,
    ("QwenVL-7B", "engagement_score"): 1, ("QwenVL-7B", "difficulty_delta"): 2,
}


def load_rq1_degrees() -> dict:
    """Read RQ1 Ferguson-retained degrees keyed by (model_label, dv)."""
    path = os.path.join(OUTPUT_DIR, "RQ1", "bic_polynomial_results.csv")
    if not os.path.exists(path):
        print(f"  [info] RQ1 degrees not found at {path}; using fallback.")
        return dict(FALLBACK_DEG)
    rq1 = pd.read_csv(path)
    dv_map = {"Engagement Score": "engagement_score",
              "Difficulty Delta": "difficulty_delta"}
    # prefer the Ferguson degree column if present, else selected_degree
    deg_col = "ferguson_degree" if "ferguson_degree" in rq1.columns \
              else "selected_degree"
    deg = {}
    for _, r in rq1.iterrows():
        dv = dv_map.get(str(r["dv"]).strip(), str(r["dv"]).strip())
        deg[(str(r["model"]).strip(), dv)] = int(r[deg_col])
    return deg


def run() -> pd.DataFrame:
    dfs  = load_all()
    deg  = load_rq1_degrees()
    rows = []

    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        df_model = dfs[model_key]
        label    = MODEL_LABELS[model_key]

        for dv in DVS:
            k = deg.get((label, dv), FALLBACK_DEG[(label, dv)])
            pair_results = []
            for treat, ref in JN_PAIRS:
                res = get_effect_curve(df_model, treat, ref, dv=dv, degree=k)
                if res is None:
                    continue
                eff, se, jn = res
                pair_results.append((treat, ref, dv, k, eff, se, jn))

                jn_pts = jn["jn_points"]
                rows.append({
                    "model":        label,
                    "dv":           dv,
                    "degree":       k,
                    "treatment":    treat,
                    "reference":    ref,
                    "n_jn_points":  len(jn_pts),
                    "jn_points":    ";".join(f"{p:.1f}" for p in jn_pts),
                    "always_sig":   jn["always_sig"],
                    "never_sig":    jn["never_sig"],
                    "region_label": jn["region_label"],
                })
                print(f"  {label:<11} {dv:<18} {treat:<9} vs {ref:<9} "
                      f"k={k}  {jn['region_label'][:46]}")

            _plot_model_dv(model_key, label, dv, pair_results)

    return pd.DataFrame(rows)


def _plot_model_dv(model_key, label, dv, pair_results):
    """One row of panels per DV for a given model: effect curve + sig shading."""
    pr = [p for p in pair_results if p[2] == dv]
    if not pr:
        return
    n = len(pr)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (treat, ref, _dv, k, eff, se, jn) in zip(axes, pr):
        mask = jn["sig_mask"]
        ax.fill_between(FIX_RANGE, -1e3, 1e3, where=mask,
                        color=COL["sig"], alpha=0.30, zorder=0)
        ax.fill_between(FIX_RANGE, eff - 1.96 * se, eff + 1.96 * se,
                        color=COL["ci"], alpha=0.55, zorder=1)
        ax.plot(FIX_RANGE, eff, color="#2166ac", lw=2, zorder=2)
        ax.axhline(0, color=COL["zero"], lw=0.9, ls="--", zorder=3)
        for jp in jn["jn_points"]:
            ax.axvline(jp, color=COL["jn"], lw=1.2, ls=":", zorder=4)

        yabs = max(abs(eff.max()), abs(eff.min()), 0.1)
        ax.set_ylim(-yabs * 1.6, yabs * 1.6)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Fixation rate (%)", fontsize=9)
        ax.set_ylabel(f"Simple effect ({dv})", fontsize=9)
        cl = CONDITION_LABELS.get(treat, treat).replace("\n", " ")
        rl = CONDITION_LABELS.get(ref, ref).replace("\n", " ")
        ax.set_title(f"{cl} vs {rl}  (k={k})", fontsize=10, fontweight="bold")

    fig.suptitle(f"Johnson-Neyman: {label} — {dv}", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    fname = f"fig_jn_{model_key.replace('.', '_')}_{dv}.png"
    fig.savefig(os.path.join(out_dir("RQ2"), fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")


if __name__ == "__main__":
    print("RQ2 | Output 2: Polynomial Johnson-Neyman\n" + "=" * 60)
    results = run()
    path = os.path.join(out_dir("RQ2"), "jn_summary.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: jn_summary.csv  ({len(results)} rows)")