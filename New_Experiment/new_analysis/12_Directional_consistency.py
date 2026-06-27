"""
12_Directional_consistency.py

Outputs:
    outputs/RQ4/direction_detail.csv     (model x condition x dv, long format)
    outputs/RQ4/direction_wide.csv       (model x dv x {M,K,GA,DG}, mirrors tab:rq4_direction)
    outputs/RQ4/direction_summary.csv    (model x dv, attenuating/amplifying classification)
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, os.path.dirname(__file__))
from config import (DATA_DIR, MODEL_FILES, MODEL_LABELS, ALL_MODELS,
                    TREATMENT_CONDITIONS, out_dir)

warnings.filterwarnings("ignore")

DVS = {"engagement_score": "Engagement Score", "difficulty_delta": "Difficulty Delta"}
DV_ABBR = {"engagement_score": "Eng.", "difficulty_delta": "Del."}
COND_ABBR = {"maori": "M", "kaumatua": "K",
            "gaze_aversion": "GA", "direct_gaze_explicit": "DG"}

FORMULAS = {
    1: "{dv} ~ fixation_rate",
    2: "{dv} ~ fixation_rate + I(fixation_rate**2)",
    3: "{dv} ~ fixation_rate + I(fixation_rate**2) + I(fixation_rate**3)",
}

FERGUSON_THRESHOLD = 0.04   # minimum practical R^2 gain (Ferguson, 2009)
LOW, HIGH = 10.0, 90.0      # interior evaluation points


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


def fit_all_degrees(dv: str, baseline: pd.DataFrame) -> dict:
    fits = {}
    for deg, formula in FORMULAS.items():
        mod = smf.ols(formula.format(dv=dv), data=baseline).fit()
        fits[deg] = {"r2": round(mod.rsquared, 4)}
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


def fit_at_degree(data: pd.DataFrame, dv: str, deg: int):
    return smf.ols(FORMULAS[deg].format(dv=dv), data=data).fit()


def fitted_rise(mod) -> float:
    """Fitted response rise yhat(HIGH) - yhat(LOW)."""
    grid = pd.DataFrame({"fixation_rate": [LOW, HIGH]})
    y_lo, y_hi = mod.predict(grid).values
    return float(y_hi - y_lo)


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows, summary_rows = [], []

    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            print(f"  [MISSING] {MODEL_FILES[model_key]}")
            continue
        data  = _load_clean(model_key)
        label = MODEL_LABELS.get(model_key, model_key)
        baseline = data[data["condition"] == "baseline"]

        for dv, dv_label in DVS.items():
            fits = fit_all_degrees(dv, baseline)
            deg  = ferguson_degree(fits)

            base_mod  = fit_at_degree(baseline, dv, deg)
            base_rise = fitted_rise(base_mod)

            changes = {}
            for cond in TREATMENT_CONDITIONS:
                cond_data = data[data["condition"] == cond]
                if cond_data.empty:
                    changes[cond] = np.nan
                    continue
                cond_mod  = fit_at_degree(cond_data, dv, deg)
                cond_rise = fitted_rise(cond_mod)
                delta     = cond_rise - base_rise
                pct       = 100.0 * delta / base_rise if base_rise != 0 else np.nan
                changes[cond] = pct
                detail_rows.append({
                    "model":      label,
                    "dv":         dv_label,
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
            if valid:
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
                    "dv":              dv_label,
                    "degree":          deg,
                    "n_conditions":    n,
                    "base_rise":       round(base_rise, 4),
                    "pct_attenuating": pct_att,
                    "pct_amplifying":  pct_amp,
                    "mean_abs_change": round(float(np.mean(np.abs(valid))), 1),
                    "direction_label": lab,
                }
                for cond in TREATMENT_CONDITIONS:
                    row[COND_ABBR[cond]] = (round(changes[cond], 1)
                                            if not np.isnan(changes.get(cond, np.nan)) else np.nan)
                summary_rows.append(row)
                print(f"  {label:<12} {dv_label:<18} d{deg}  "
                      f"att={pct_att:.0f}% amp={pct_amp:.0f}%  "
                      f"base_rise={base_rise:+.3f}  [{lab}]")

    detail_df  = pd.DataFrame(detail_rows)
    summary_df = pd.DataFrame(summary_rows)

    # Wide pivot mirroring tab:rq4_direction exactly: Model, DV, M, K, GA, DG
    wide = detail_df.pivot_table(index=["model", "dv"], columns="condition",
                                 values="pct_change", aggfunc="first").reset_index()
    wide["dv"] = wide["dv"].map({v: DV_ABBR[k] for k, v in DVS.items()})
    wide = wide.rename(columns=COND_ABBR)
    col_order = ["model", "dv"] + [COND_ABBR[c] for c in TREATMENT_CONDITIONS]
    wide = wide[[c for c in col_order if c in wide.columns]]

    return detail_df, wide, summary_df


if __name__ == "__main__":
    print("12 | RQ4: Directional Consistency\n" + "=" * 60)
    detail, wide, summary = run()

    p1 = os.path.join(out_dir("RQ4"), "direction_detail.csv")
    p2 = os.path.join(out_dir("RQ4"), "direction_wide.csv")
    p3 = os.path.join(out_dir("RQ4"), "direction_summary.csv")
    detail.to_csv(p1, index=False)
    wide.to_csv(p2, index=False)
    summary.to_csv(p3, index=False)

    print("\nWide (mirrors tab:rq4_direction):")
    print(wide.to_string(index=False))
    print(f"\nSaved: {p1}  ({len(detail)} rows)")
    print(f"Saved: {p2}  ({len(wide)} rows)")
    print(f"Saved: {p3}  ({len(summary)} rows)")
