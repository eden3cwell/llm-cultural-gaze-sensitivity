"""
03_BIC_Fit.py

Output: outputs/RQ1/bic_fit.csv
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

FORMULAS = {
    1: "{dv} ~ fixation_rate",
    2: "{dv} ~ fixation_rate + I(fixation_rate**2)",
    3: "{dv} ~ fixation_rate + I(fixation_rate**2) + I(fixation_rate**3)",
}


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


def fit_all_degrees(dv: str, baseline: pd.DataFrame) -> dict:
    fits = {}
    for deg, formula in FORMULAS.items():
        mod = smf.ols(formula.format(dv=dv), data=baseline).fit()
        fits[deg] = {"model": mod, "r2": round(mod.rsquared, 4), "bic": round(mod.bic, 2)}
    return fits


def bic_degree(fits: dict) -> int:
    return min(fits, key=lambda d: fits[d]["bic"])


def classify_shape(deg: int, mod, x_min: float, x_max: float) -> str:
    """Shape from the fitted coefficients.

    deg 1: Linear (or Flat if the slope is ~0).
    deg 2: Quadratic, labelled by concavity.
    deg 3: a cubic has an inflection point at x* = -b2/(3*b3). If x* falls
        inside the observed fixation-rate range, the curve genuinely changes
        concavity within the data and is Sigmoidal, not uniformly
        concave-up/down. If x* falls outside the range, the cubic doesn't
        bend within the data and is reported as Quadratic-like concavity
        using the quadratic term's sign.
    """
    if deg == 1:
        return "Linear" if abs(mod.params["fixation_rate"]) > 1e-6 else "Flat"

    b2 = mod.params["I(fixation_rate ** 2)"]
    if deg == 2:
        return "Quadratic (Concave-up)" if b2 > 0 else "Quadratic (Concave-down)"

    b3 = mod.params["I(fixation_rate ** 3)"]
    if abs(b3) < 1e-12:
        return "Quadratic (Concave-up)" if b2 > 0 else "Quadratic (Concave-down)"
    x_inflect = -b2 / (3 * b3)
    if x_min <= x_inflect <= x_max:
        return "Sigmoidal"
    return "Quadratic (Concave-up)" if b2 > 0 else "Quadratic (Concave-down)"


def run() -> pd.DataFrame:
    rows = []
    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            print(f"  [MISSING] {MODEL_FILES[model_key]}")
            continue
        baseline = _load_baseline(model_key)
        label    = MODEL_LABELS.get(model_key, model_key)

        x_min, x_max = baseline["fixation_rate"].min(), baseline["fixation_rate"].max()

        for dv, dv_label in DVS.items():
            fits    = fit_all_degrees(dv, baseline)
            bic_deg = bic_degree(fits)
            shape   = classify_shape(bic_deg, fits[bic_deg]["model"], x_min, x_max)
            delta_r2 = round(fits[bic_deg]["r2"] - fits[1]["r2"], 4) if bic_deg > 1 else 0.0

            rows.append({
                "model":              label,
                "dv":                 dv_label,
                "n_obs":              len(baseline),
                "deg1_r2":            fits[1]["r2"],
                "deg2_r2":            fits[2]["r2"],
                "deg3_r2":            fits[3]["r2"],
                "deg1_bic":           fits[1]["bic"],
                "deg2_bic":           fits[2]["bic"],
                "deg3_bic":           fits[3]["bic"],
                "bic_degree":         bic_deg,
                "bic_r2":             fits[bic_deg]["r2"],
                "delta_r2_vs_linear": delta_r2,
                "shape":              shape,
            })
            print(f"  {label:<12} {dv_label:<16} "
                  f"BIC=d{bic_deg}  R2={fits[bic_deg]['r2']:.3f}  "
                  f"deltaR2={delta_r2:.3f}  {shape}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("03 | BIC Polynomial Fit\n" + "=" * 60)
    results = run()
    path = os.path.join(out_dir("RQ1"), "bic_fit.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")
