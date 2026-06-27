"""
10_Explicit_DifferenceDirection.py

Outputs:
    outputs/RQ3/binned_mean_diff.csv
    outputs/RQ3/overall_aggregate.csv
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (DATA_DIR, MODEL_FILES, MODEL_LABELS, ALL_MODELS,
                    EXPLICIT_CONDITIONS, BIN_EDGES, BIN_LABELS, out_dir)

warnings.filterwarnings("ignore")

DVS = ["engagement_score", "difficulty_delta"]


def _load_clean(model_key: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[df["parse_error"].isna() | (df["parse_error"].astype(str).isin(["", "nan", "None"]))]
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["engagement_score", "difficulty_delta", "fixation_rate", "condition"])
    df = df[(df.engagement_score >= 1) & (df.engagement_score <= 10)
            & (df.difficulty_delta >= -1) & (df.difficulty_delta <= 1)]
    return df


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD standardised mean difference. No standard library function
    computes this directly, so it stays hand-derived."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    psd = np.sqrt(((n_a - 1) * va + (n_b - 1) * vb) / (n_a + n_b - 2))
    return 0.0 if psd == 0 else float((a.mean() - b.mean()) / psd)


def run_binned() -> pd.DataFrame:
    rows = []
    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            print(f"  [MISSING] {MODEL_FILES[model_key]}")
            continue
        dm = _load_clean(model_key)
        label = MODEL_LABELS.get(model_key, model_key)
        dm["bin"] = pd.cut(dm["fixation_rate"], bins=BIN_EDGES,
                           labels=BIN_LABELS, right=False)
        baseline = dm[dm["condition"] == "baseline"]

        for cond in EXPLICIT_CONDITIONS:
            treat = dm[dm["condition"] == cond]
            for b in BIN_LABELS:
                bl = baseline[baseline["bin"] == b]
                tr = treat[treat["bin"] == b]
                row = {"model": label, "condition": cond, "bin": b,
                       "n_baseline": len(bl), "n_treatment": len(tr)}
                for dv in DVS:
                    if len(bl) and len(tr):
                        row[f"meandiff_{dv}"] = round(
                            float(tr[dv].mean() - bl[dv].mean()), 4)
                    else:
                        row[f"meandiff_{dv}"] = np.nan
                rows.append(row)
                print(f"  {label:<12} {cond:<22} {b:<10} "
                      f"dEng={row['meandiff_engagement_score']:>+7}  "
                      f"dDelta={row['meandiff_difficulty_delta']:>+7}")

    return pd.DataFrame(rows)


def run_overall() -> pd.DataFrame:
    rows = []
    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            continue
        dm = _load_clean(model_key)
        label = MODEL_LABELS.get(model_key, model_key)
        baseline = dm[dm["condition"] == "baseline"]

        for cond in EXPLICIT_CONDITIONS:
            treat = dm[dm["condition"] == cond]
            if len(treat) == 0:
                continue
            row = {"model": label, "condition": cond}
            for dv in DVS:
                row[f"meandiff_{dv}"] = round(
                    float(treat[dv].mean() - baseline[dv].mean()), 4)
                row[f"d_{dv}"] = round(
                    cohens_d(treat[dv].values, baseline[dv].values), 4)
            rows.append(row)
            print(f"  {label:<12} {cond:<22} "
                  f"dEng={row['meandiff_engagement_score']:>+7} (d={row['d_engagement_score']:+.3f})  "
                  f"dDelta={row['meandiff_difficulty_delta']:>+7} (d={row['d_difficulty_delta']:+.3f})")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("10 | RQ3: Binned + Overall Mean Differences (Explicit)\n" + "=" * 60)

    binned = run_binned()
    pb = os.path.join(out_dir("RQ3"), "binned_mean_diff.csv")
    binned.to_csv(pb, index=False)
    print(f"\nSaved: {pb}  ({len(binned)} rows)")

    overall = run_overall()
    po = os.path.join(out_dir("RQ3"), "overall_aggregate.csv")
    overall.to_csv(po, index=False)
    print(f"Saved: {po}  ({len(overall)} rows)")
