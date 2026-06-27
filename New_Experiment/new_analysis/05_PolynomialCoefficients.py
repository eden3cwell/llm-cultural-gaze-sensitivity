"""
05_PolynomialCoefficients.py

Output: outputs/RQ1/ferguson_coefficients.csv
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
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

FERGUSON_THRESHOLD = 0.04   # minimum practical R^2 gain (Ferguson, 2009)


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
        fits[deg] = {"model": mod, "r2": round(mod.rsquared, 4)}
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


def classify_shape(deg: int, mod, x_min: float, x_max: float) -> str:
    """Shape from the fitted coefficients (identical to 04_FergusonSelection.py)."""
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


def coefficients(deg: int, mod) -> tuple[float, float, float, float]:
    """beta0..beta3, with terms above the selected degree left as NaN."""
    b0 = mod.params["Intercept"]
    b1 = mod.params["fixation_rate"] if deg >= 1 else np.nan
    b2 = mod.params["I(fixation_rate ** 2)"] if deg >= 2 else np.nan
    b3 = mod.params["I(fixation_rate ** 3)"] if deg >= 3 else np.nan
    return b0, b1, b2, b3


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
            fits     = fit_all_degrees(dv, baseline)
            ferg_deg = ferguson_degree(fits)
            mod      = fits[ferg_deg]["model"]
            shape    = classify_shape(ferg_deg, mod, x_min, x_max)
            b0, b1, b2, b3 = coefficients(ferg_deg, mod)

            rows.append({
                "model":           label,
                "dv":              dv_label,
                "n_obs":           len(baseline),
                "ferguson_degree": ferg_deg,
                "shape":           shape,
                "beta0":           round(b0, 6),
                "beta1":           round(b1, 6) if not np.isnan(b1) else np.nan,
                "beta2":           round(b2, 8) if not np.isnan(b2) else np.nan,
                "beta3":           round(b3, 10) if not np.isnan(b3) else np.nan,
                "r2":              fits[ferg_deg]["r2"],
            })
            print(f"  {label:<12} {dv_label:<16} d{ferg_deg}  "
                  f"[{b0:+.4f}, {b1:+.4f}, "
                  f"{'' if np.isnan(b2) else f'{b2:+.6f}'}, "
                  f"{'' if np.isnan(b3) else f'{b3:+.2e}'}]  "
                  f"R2={fits[ferg_deg]['r2']:.3f}  {shape}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("05 | Polynomial Coefficients (Ferguson-selected degree)\n" + "=" * 60)
    results = run()
    path = os.path.join(out_dir("RQ1"), "ferguson_coefficients.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")
