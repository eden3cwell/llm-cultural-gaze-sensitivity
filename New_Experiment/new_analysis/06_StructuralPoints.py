"""
06_StructuralPoints.py

Output: outputs/summary/structural_points.csv
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, MODEL_FILES, MODEL_LABELS, ALL_MODELS, CONDITIONS, out_dir

warnings.filterwarnings("ignore")

DV = "difficulty_delta"

FORMULAS = {
    1: "{dv} ~ fixation_rate",
    2: "{dv} ~ fixation_rate + I(fixation_rate**2)",
    3: "{dv} ~ fixation_rate + I(fixation_rate**2) + I(fixation_rate**3)",
}

FERGUSON_THRESHOLD = 0.04   # minimum practical R^2 gain (Ferguson, 2009)
X_RANGE = np.linspace(0, 100, 2000)


def _load_clean(model_key: str) -> pd.DataFrame:
    """All conditions, cleaned (filtering to a specific condition is the
    caller's job)."""
    path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[df["parse_error"].isna() | (df["parse_error"].astype(str).isin(["", "nan", "None"]))]
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["engagement_score", "difficulty_delta", "fixation_rate", "condition"])
    df = df[(df.engagement_score >= 1) & (df.engagement_score <= 10)
            & (df.difficulty_delta >= -1) & (df.difficulty_delta <= 1)]
    return df


def fit_all_degrees(data: pd.DataFrame) -> dict:
    fits = {}
    for deg, formula in FORMULAS.items():
        mod = smf.ols(formula.format(dv=DV), data=data).fit()
        fits[deg] = {"model": mod, "r2": round(mod.rsquared, 4)}
    return fits


def fit_at_degree(data: pd.DataFrame, deg: int):
    return smf.ols(FORMULAS[deg].format(dv=DV), data=data).fit()


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


def predict_curve(mod) -> np.ndarray:
    df_grid = pd.DataFrame({"fixation_rate": X_RANGE})
    return mod.predict(df_grid).values


def zero_crossings(y: np.ndarray, x: np.ndarray = X_RANGE) -> list[tuple[float, int]]:
    """Return [(x, direction)] where y crosses 0; direction +1 = upward."""
    out = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:
            xc = x[i] + (0 - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
            out.append((float(xc), 1 if y[i + 1] > y[i] else -1))
    return out


def structural_points(y: np.ndarray) -> tuple[float | None, float, float | None]:
    cross  = zero_crossings(y)
    r_peak = float(X_RANGE[int(np.argmax(y))])
    r_plus = next((x for x, d in cross if d > 0), None)
    if r_plus is None and cross:
        r_plus = cross[0][0]
    r_minus = next((x for x, d in cross if d < 0 and x > r_peak), None)
    return r_plus, r_peak, r_minus


def run() -> pd.DataFrame:
    rows = []
    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            print(f"  [MISSING] {MODEL_FILES[model_key]}")
            continue
        data  = _load_clean(model_key)
        label = MODEL_LABELS.get(model_key, model_key)

        baseline = data[data["condition"] == "baseline"]
        x_min, x_max = baseline["fixation_rate"].min(), baseline["fixation_rate"].max()

        fits     = fit_all_degrees(baseline)
        ferg_deg = ferguson_degree(fits)

        bl_rp = bl_rpeak = bl_rm = None  # set once baseline is processed

        for cond in CONDITIONS:
            cond_data = data[data["condition"] == cond]
            if cond_data.empty:
                continue
            mod   = fit_at_degree(cond_data, ferg_deg)
            shape = classify_shape(ferg_deg, mod, x_min, x_max)

            y_fit = predict_curve(mod)
            rp, rpeak, rm = structural_points(y_fit)

            if cond == "baseline":
                bl_rp, bl_rpeak, bl_rm = rp, rpeak, rm

            shift_rp    = round(rp - bl_rp, 2) if (rp is not None and bl_rp is not None) else np.nan
            shift_rpeak = round(rpeak - bl_rpeak, 2) if bl_rpeak is not None else np.nan
            shift_rm    = round(rm - bl_rm, 2) if (rm is not None and bl_rm is not None) else np.nan

            rows.append({
                "model":           label,
                "condition":       cond,
                "n_obs":           len(cond_data),
                "ferguson_degree": ferg_deg,
                "shape":           shape,
                "r_plus":          round(rp, 2) if rp is not None else np.nan,
                "r_peak":          round(rpeak, 2),
                "r_minus":         round(rm, 2) if rm is not None else np.nan,
                "delta_min":       round(float(y_fit.min()), 4),
                "delta_max":       round(float(y_fit.max()), 4),
                "shift_r_plus":    shift_rp,
                "shift_r_peak":    shift_rpeak,
                "shift_r_minus":   shift_rm,
            })
            print(f"  {label:<12} {cond:<22} d{ferg_deg} {shape:<22} "
                  f"r+={rows[-1]['r_plus']}  r_peak={rows[-1]['r_peak']}  "
                  f"r-={rows[-1]['r_minus']}  shift_peak={shift_rpeak}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("06 | Structural Points (all conditions, difficulty_delta)\n" + "=" * 60)
    results = run()
    path = os.path.join(out_dir("summary"), "structural_points.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")
