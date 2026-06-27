"""
02_OLS_Slope_and_WithinCellVariance.py

Output: outputs/RQ1/slope_and_variance.csv
"""

from __future__ import annotations

import os
import sys
import warnings
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, MODEL_FILES, MODEL_LABELS, ALL_MODELS, out_dir

warnings.filterwarnings("ignore")

DVS = {"engagement_score": "Engagement", "difficulty_delta": "Difficulty Delta"}


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


def within_cell_variance(baseline: pd.DataFrame, dv: str) -> float:
    """Mean across fixation cells of run-to-run sample variance (ddof=1)."""
    return float(baseline.groupby("fixation_rate")[dv].var(ddof=1).mean())


def run() -> pd.DataFrame:
    rows = []
    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            print(f"  [MISSING] {MODEL_FILES[model_key]}")
            continue
        baseline = _load_baseline(model_key)
        label = MODEL_LABELS.get(model_key, model_key)
        n_obs = len(baseline)

        for dv, dv_label in DVS.items():
            mod = smf.ols(f"{dv} ~ fixation_rate", data=baseline).fit()
            b0, se_b0 = mod.params["Intercept"], mod.bse["Intercept"]
            b1, se_b1 = mod.params["fixation_rate"], mod.bse["fixation_rate"]
            t_b1, p_b1, r2_lin = mod.tvalues["fixation_rate"], mod.pvalues["fixation_rate"], mod.rsquared
            sigma2 = within_cell_variance(baseline, dv)

            rows.append({
                "model": label, "dv": dv_label, "n_obs": n_obs,
                "beta0": round(b0, 6), "beta0_se": round(se_b0, 6),
                "beta1": round(b1, 6), "beta1_se": round(se_b1, 6),
                "beta1_t": round(t_b1, 3), "beta1_p": round(p_b1, 6),
                "r2_linear": round(r2_lin, 4),
                "sig": "***" if p_b1 < .001 else ("**" if p_b1 < .01
                       else ("*" if p_b1 < .05 else "ns")),
                "sigma2": round(sigma2, 4),
            })
            print(f"  {label:<12} {dv_label:<16} beta0={b0:+.4f}  beta1={b1:+.4f}  "
                  f"R2(lin)={r2_lin:.3f}  sigma2={sigma2:.4f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("02 | OLS Slope + Within-Cell Variance\n" + "=" * 60)
    results = run()
    path = os.path.join(out_dir("RQ1"), "AppendixTableII:slope_and_variance.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")
