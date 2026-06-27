"""
RQ1 | Output 1: BIC + Ferguson polynomial fit + within-cell variance.

For each model, on the FULL RAW baseline observations (n ~ 1010):
  - fits polynomial degrees 1-3, reports R^2 per degree
  - selects degree by BIC (statistical criterion, retained for comparison)
  - selects degree by a two-condition Ferguson practical screen:
        (a) the chosen degree gains >= FERGUSON_THRESHOLD R^2 over linear, and
        (b) every term retained beyond the first added >= FERGUSON_THRESHOLD
            over the degree below it.
    A model stays linear if no degree clears (a). This correctly retains
    cubic-dominant (sigmoidal) shapes whose quadratic component alone is weak,
    while rejecting higher-order terms that do not materially improve fit.
  - reports the Ferguson degree, delta-R^2 over linear, the marginal gain of
    the last retained term, curve shape, and within-cell variance sigma^2.

The Ferguson degree is the one reported in the thesis tables.

Output: outputs/new_experiment/RQ1/bic_polynomial_results.csv
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import INFERENTIAL_MODELS, MODEL_LABELS, out_dir
from loaders import load_all

warnings.filterwarnings("ignore")

DVS = {
    "engagement_score": "Engagement Score",
    "difficulty_delta":  "Difficulty Delta",
}

FERGUSON_THRESHOLD = 0.04   # minimum practical R^2 gain (Ferguson, 2009)


def bic(n: int, sse: float, k: int) -> float:
    return n * np.log(sse / n) + k * np.log(n)


def fit_all_degrees(x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> dict:
    n = len(x)
    results = {}
    for deg in range(1, max_degree + 1):
        coeffs  = np.polyfit(x, y, deg)
        y_pred  = np.polyval(coeffs, x)
        sse     = float(np.sum((y - y_pred) ** 2))
        sst     = float(np.sum((y - y.mean()) ** 2))
        r2      = 1.0 - sse / sst if sst > 0 else 0.0
        bic_val = bic(n, sse, deg + 1)
        results[deg] = {"coeffs": coeffs, "r2": round(r2, 4),
                        "bic": round(bic_val, 2)}
    return results


def bic_degree(fits: dict) -> int:
    return min(fits, key=lambda d: fits[d]["bic"])


def ferguson_degree(fits: dict, threshold: float = FERGUSON_THRESHOLD) -> int:
    """Two-condition practical screen.

    (a) Find the LOWEST degree whose R^2 gain over linear is >= threshold.
        If none, the model is linear (return 1).
    (b) From that base, extend upward only while each further term adds
        >= threshold over the current accepted degree.
    """
    # (a) lowest degree beating linear by >= threshold
    base = None
    for deg in range(2, max(fits) + 1):
        if fits[deg]["r2"] - fits[1]["r2"] >= threshold:
            base = deg
            break
    if base is None:
        return 1

    # (b) extend only while each added term independently earns >= threshold
    accepted = base
    for deg in range(base + 1, max(fits) + 1):
        if fits[deg]["r2"] - fits[accepted]["r2"] >= threshold:
            accepted = deg
        else:
            break
    return accepted


def classify_shape(deg: int, coeffs: np.ndarray) -> str:
    """Shape from leading coefficients (polyfit returns highest power first).

    deg 1: linear.
    deg 2: sign of quadratic term (coeffs[0]).
    deg 3: sign of quadratic term (coeffs[1]) as dominant concavity.
    """
    if deg == 1:
        return "Linear" if abs(coeffs[0]) > 1e-6 else "Flat"
    if deg == 2:
        return "Concave-up" if coeffs[0] > 0 else "Concave-down"
    return "Concave-up" if coeffs[1] > 0 else "Concave-down"


def within_cell_variance(baseline: pd.DataFrame, dv: str) -> float:
    """Mean across fixation cells of run-to-run sample variance (ddof=1)."""
    return float(baseline.groupby("fixation_rate")[dv].var(ddof=1).mean())


def run() -> pd.DataFrame:
    dfs = load_all()
    rows = []

    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        baseline = dfs[model_key][dfs[model_key]["condition"] == "baseline"]
        label    = MODEL_LABELS[model_key]

        for dv, dv_label in DVS.items():
            x = baseline["fixation_rate"].values.astype(float)
            y = baseline[dv].values.astype(float)

            fits     = fit_all_degrees(x, y)
            bic_deg  = bic_degree(fits)
            ferg_deg = ferguson_degree(fits)

            delta_r2 = round(fits[ferg_deg]["r2"] - fits[1]["r2"], 4) \
                       if ferg_deg > 1 else 0.0
            last_gain = round(fits[ferg_deg]["r2"] - fits[ferg_deg - 1]["r2"], 4) \
                        if ferg_deg > 1 else 0.0

            shape  = classify_shape(ferg_deg, fits[ferg_deg]["coeffs"])
            sigma2 = round(within_cell_variance(baseline, dv), 4)
            agree  = "yes" if bic_deg == ferg_deg else "NO"

            rows.append({
                "model":              label,
                "dv":                 dv_label,
                "n_obs":              len(x),
                "sigma2":             sigma2,
                "deg1_r2":            fits[1]["r2"],
                "deg2_r2":            fits[2]["r2"],
                "deg3_r2":            fits[3]["r2"],
                "bic_degree":         bic_deg,
                "ferguson_degree":    ferg_deg,
                "bic_ferguson_agree": agree,
                "ferguson_r2":        fits[ferg_deg]["r2"],
                "delta_r2_vs_linear": delta_r2,
                "last_term_gain":     last_gain,
                "shape":              shape,
            })
            flag = "" if agree == "yes" else "   <-- BIC/Ferguson differ"
            print(f"  {label:<12} {dv_label:<18} "
                  f"BIC=d{bic_deg}  Ferg=d{ferg_deg}  "
                  f"ΔR²={delta_r2:.3f}  last+={last_gain:.3f}  "
                  f"{shape}{flag}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("RQ1 | Output 1: BIC + Ferguson Polynomial Fit\n" + "=" * 60)
    results = run()
    path = os.path.join(out_dir("RQ1"), "bic_polynomial_results.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")