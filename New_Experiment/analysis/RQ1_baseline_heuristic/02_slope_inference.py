"""
RQ1 | Output 2: OLS slope inference — baseline, full raw data.

Reports the DEGREE-1 linear slope (beta_1) as the overall-direction
statistic for each model and DV, fitted on the full raw baseline data.
This is the comparable "average trend" measure: a positive beta_1 on
engagement operationalises the WEIRD high-fixation heuristic.

The BIC-selected polynomial degree (also computed on raw data, matching
Output 1) is reported in a separate column as the SHAPE layer, so the
direction statistic and the curvature statistic never conflict.

Note: because model output is near-deterministic, the p-values are
effectively zero by construction and are retained only for completeness;
inference in the thesis is effect-size-led, not p-led.

Output: outputs/new_experiment/RQ1/slope_inference_table.csv
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import INFERENTIAL_MODELS, MODEL_LABELS, out_dir
from loaders import load_all

warnings.filterwarnings("ignore")

DVS = ["engagement_score", "difficulty_delta"]


def bic_degree_raw(x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> int:
    """Return BIC-selected polynomial degree on the raw observations.

    Matches the selection rule used in Output 1 so the reported degree
    is identical across both tables.
    """
    n = len(x)
    best_bic, best_deg = np.inf, 1
    for deg in range(1, max_degree + 1):
        coeffs = np.polyfit(x, y, deg)
        sse    = float(np.sum((y - np.polyval(coeffs, x)) ** 2))
        bic    = n * np.log(sse / n) + (deg + 1) * np.log(n)
        if bic < best_bic:
            best_bic, best_deg = bic, deg
    return best_deg


def run() -> pd.DataFrame:
    dfs  = load_all()
    rows = []

    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        df_model = dfs[model_key]
        baseline = df_model[df_model["condition"] == "baseline"].copy()
        label    = MODEL_LABELS[model_key]
        n_obs    = len(baseline)

        for dv in DVS:
            x = baseline["fixation_rate"].values.astype(float)
            y = baseline[dv].values.astype(float)

            # --- SHAPE layer: BIC degree on raw data (for reporting only) ---
            bic_deg = bic_degree_raw(x, y)

            # --- DIRECTION layer: always degree-1 slope ---
            mod   = smf.ols(f"{dv} ~ fixation_rate", data=baseline).fit()
            b1    = mod.params["fixation_rate"]
            se_b1 = mod.bse["fixation_rate"]
            t_b1  = mod.tvalues["fixation_rate"]
            p_b1  = mod.pvalues["fixation_rate"]
            r2_lin = mod.rsquared

            rows.append({
                "model":       label,
                "dv":          dv,
                "n_obs":       n_obs,
                "bic_degree":  bic_deg,          # shape layer (raw-data BIC)
                "beta1":       round(b1,  6),     # direction: degree-1 slope
                "beta1_se":    round(se_b1, 6),
                "beta1_t":     round(t_b1,  3),
                "beta1_p":     round(p_b1,  6),
                "r2_linear":   round(r2_lin, 4),
                "sig":         "***" if p_b1 < .001 else ("**" if p_b1 < .01
                                else ("*" if p_b1 < .05 else "ns")),
            })
            print(f"  {label:<12} {dv:<22} β₁={b1:+.4f}  "
                  f"BIC-deg={bic_deg}  R²(lin)={r2_lin:.3f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("RQ1 | Output 2: OLS Slope Inference (degree-1 direction)\n" + "=" * 60)
    results = run()
    path = os.path.join(out_dir("RQ1"), "slope_inference_table.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")