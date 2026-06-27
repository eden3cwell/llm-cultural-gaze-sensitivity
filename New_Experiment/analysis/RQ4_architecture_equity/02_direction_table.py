"""
RQ4 | Output 2: Direction table (rise-change basis).

For each inferential model, fits the RQ1 Ferguson-retained polynomial to
baseline and to each treatment condition (both DVs), and computes the change in
fitted response amplitude relative to baseline:

    rise(c)    = yhat_c(90) - yhat_c(10)              # fitted rise across fixation
    pct_change = 100 * (rise(c) - rise(bl)) / rise(bl)

A negative pct_change means the condition ATTENUATED the heuristic (flattened the
response, reducing reliance on fixation); positive means it AMPLIFIED it
(steepened). This replaces the earlier Cohen's-d-sign measure, which indexed the
LEVEL of the output rather than the SLOPE of the heuristic and therefore
mislabelled level shifts as WEIRD-amplification.

The rise is taken at interior points (10%, 90%) of the fitted polynomial to
respect curvature and avoid linear extrapolation. pct_change is proportional to
each model's own baseline amplitude; base_rise is retained for context and is
unstable where small.

Outputs:
    outputs/new_experiment/RQ4/direction_table.csv     (model x condition x dv)
    outputs/new_experiment/RQ4/direction_summary.csv   (model x dv)
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import (INFERENTIAL_MODELS, TREATMENT_CONDITIONS,
                    MODEL_LABELS, OUTPUT_DIR, out_dir)
from loaders import load_all, cell_means

warnings.filterwarnings("ignore")

DVS = ["engagement_score", "difficulty_delta"]
LOW, HIGH = 10.0, 90.0          # interior evaluation points


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


def fitted_rise(df, condition, dv, degree):
    """Fitted response rise yhat(HIGH) - yhat(LOW) for one condition."""
    sub = df[df["condition"] == condition]
    cm  = cell_means(sub, dv)
    x   = cm["fixation_rate"].values.astype(float)
    y   = cm[f"{dv}_mean"].values
    c   = np.polyfit(x, y, degree)
    return float(np.polyval(c, HIGH) - np.polyval(c, LOW))


def run():
    dfs        = load_all()
    FERGUSON   = load_rq1_degrees()
    detail_rows, summary_rows = [], []

    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        df_model = dfs[model_key]
        label    = MODEL_LABELS[model_key]

        for dv in DVS:
            bl  = df_model[df_model["condition"] == "baseline"]
            cm  = cell_means(bl, dv)
            deg = FERGUSON.get(
                (label, dv),
                bic_degree(cm["fixation_rate"].values.astype(float),
                           cm[f"{dv}_mean"].values))
            base_rise = fitted_rise(df_model, "baseline", dv, deg)

            changes = {}
            for cond in TREATMENT_CONDITIONS:
                if len(df_model[df_model["condition"] == cond]) == 0:
                    changes[cond] = np.nan
                    continue
                cond_rise = fitted_rise(df_model, cond, dv, deg)
                delta     = cond_rise - base_rise
                pct       = 100.0 * delta / base_rise if base_rise != 0 else np.nan
                changes[cond] = pct
                detail_rows.append({
                    "model":      label,
                    "dv":         dv,
                    "condition":  cond,
                    "degree":     deg,
                    "base_rise":  round(base_rise, 4),
                    "cond_rise":  round(cond_rise, 4),
                    "delta_rise": round(delta, 4),
                    "pct_change": round(pct, 1) if not np.isnan(pct) else np.nan,
                    "direction":  ("attenuating" if delta < 0
                                   else "amplifying" if delta > 0 else "none"),
                })

            valid = [v for v in changes.values() if not np.isnan(v)]
            if not valid:
                continue
            n     = len(valid)
            n_att = sum(1 for v in valid if v < 0)
            n_amp = sum(1 for v in valid if v > 0)
            pct_att = round(100 * n_att / n, 1)
            pct_amp = round(100 * n_amp / n, 1)
            if pct_att == 100:
                lab = "Consistently attenuating"
            elif pct_amp == 100:
                lab = "Consistently amplifying"
            elif pct_att >= 75:
                lab = "Predominantly attenuating"
            elif pct_amp >= 75:
                lab = "Predominantly amplifying"
            else:
                lab = "Mixed directional response"

            row = {
                "model":           label,
                "dv":              dv,
                "n_conditions":    n,
                "base_rise":       round(base_rise, 4),
                "pct_attenuating": pct_att,
                "pct_amplifying":  pct_amp,
                "mean_abs_change": round(float(np.mean(np.abs(valid))), 1),
                "direction_label": lab,
            }
            for cond in TREATMENT_CONDITIONS:
                row[f"change_{cond}"] = (round(changes[cond], 1)
                                        if not np.isnan(changes.get(cond, np.nan)) else np.nan)
            summary_rows.append(row)
            print(f"  {label:<12} {dv:<22} att={pct_att:.0f}% amp={pct_amp:.0f}% "
                  f"base_rise={base_rise:+.3f}  [{lab}]")

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


if __name__ == "__main__":
    print("RQ4 | Output 2: Direction Table (rise-change)\n" + "=" * 60)
    detail_df, summary_df = run()
    p1 = os.path.join(out_dir("RQ4"), "direction_table.csv")
    p2 = os.path.join(out_dir("RQ4"), "direction_summary.csv")
    detail_df.to_csv(p1, index=False)
    summary_df.to_csv(p2, index=False)
    print(f"\nSaved: {p1}  ({len(detail_df)} rows)")
    print(f"Saved: {p2}  ({len(summary_df)} rows)")