"""
03b_poly_coefficients.py  —  MISSING-GENERATOR FILL (additive; 01/02 untouched)

Emits the per-model polynomial COEFFICIENTS that the appendix table
`tab:app_polycoef` reports but no committed script writes (01_polynomial_fit.py
computes the coefficients then discards them).

For each model x DV, on the raw baseline observations (n ~ 1010), it reports the
linear/quadratic/cubic coefficients (low -> high power) of BOTH:
    - the BIC-selected fit, and
    - the Ferguson-retained fit (the shape the thesis reports elsewhere),
together with each fit's R^2.

Why both: the current `tab:app_polycoef` is captioned "BIC-selected" but for
Qwen-14B / Mistral engagement it actually prints the Ferguson (degree-2)
coefficients, whereas BIC selects degree 3 for those cells. Emitting both
columns lets the appendix be made internally consistent under either convention.

Self-scoped to the seven inferential models (unaffected by the 32B config drift).

Output: outputs/new_experiment/RQ1/poly_coefficients.csv
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import DATA_DIR, out_dir

ROSTER = [
    ("Llama-3.2-3B", "Llama-3B"), ("Llama-3.1-8B", "Llama-8B"),
    ("Llama-3.2-11B", "Llama-11B"), ("Mistral-7B", "Mistral-7B"),
    ("Qwen2.5-7B", "Qwen-7B"), ("Qwen2.5-14B", "Qwen-14B"),
    ("Qwen2.5-VL-7B", "QwenVL-7B"),
]
DVS = {"engagement_score": "Engagement", "difficulty_delta": "Difficulty Delta"}
FERGUSON_THRESHOLD = 0.04


def _clean(df):
    df = df[df["parse_error"].isna() | (df["parse_error"].astype(str).isin(["", "nan", "None"]))].copy()
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["engagement_score", "difficulty_delta", "fixation_rate", "condition"])
    return df[(df.engagement_score >= 1) & (df.engagement_score <= 10)
              & (df.difficulty_delta >= -1) & (df.difficulty_delta <= 1)]


def _fits(x, y):
    out = {}
    n = len(x)
    for d in (1, 2, 3):
        c = np.polyfit(x, y, d)
        sse = float(np.sum((y - np.polyval(c, x)) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        out[d] = {"coeffs": c, "r2": 1 - sse / sst if sst else 0.0,
                  "bic": n * np.log(sse / n) + (d + 1) * np.log(n)}
    return out


def _bic_degree(f):
    return min(f, key=lambda d: f[d]["bic"])


def _ferguson_degree(f, thr=FERGUSON_THRESHOLD):
    base = None
    for d in range(2, max(f) + 1):
        if f[d]["r2"] - f[1]["r2"] >= thr:
            base = d
            break
    if base is None:
        return 1
    acc = base
    for d in range(base + 1, max(f) + 1):
        if f[d]["r2"] - f[acc]["r2"] >= thr:
            acc = d
        else:
            break
    return acc


def _coef_row(coeffs):
    """polyfit returns high->low power; return (lin, quad, cub) low->high."""
    lo = coeffs[::-1]
    lin = lo[1] if len(lo) > 1 else 0.0
    quad = lo[2] if len(lo) > 2 else 0.0
    cub = lo[3] if len(lo) > 3 else 0.0
    return lin, quad, cub


def run():
    rows = []
    for stem, label in ROSTER:
        df = _clean(pd.read_csv(os.path.join(DATA_DIR, f"{stem}.csv")))
        b = df[df.condition == "baseline"]
        for dv, dv_label in DVS.items():
            x = b.fixation_rate.values.astype(float)
            y = b[dv].values.astype(float)
            f = _fits(x, y)
            bdeg, fdeg = _bic_degree(f), _ferguson_degree(f)
            bl, bq, bc = _coef_row(f[bdeg]["coeffs"])
            fl, fq, fc = _coef_row(f[fdeg]["coeffs"])
            rows.append({
                "model": label, "dv": dv_label,
                "bic_degree": bdeg, "bic_b1": round(bl, 5), "bic_b2": round(bq, 6),
                "bic_b3": round(bc, 8), "bic_r2": round(f[bdeg]["r2"], 4),
                "ferguson_degree": fdeg, "ferg_b1": round(fl, 5), "ferg_b2": round(fq, 6),
                "ferg_b3": round(fc, 8), "ferg_r2": round(f[fdeg]["r2"], 4),
            })
            print(f"  {label:<11}{dv_label:<16} BIC d{bdeg} "
                  f"[{bl:+.5f},{bq:+.6f},{bc:+.2e}] R²={f[bdeg]['r2']:.3f} | "
                  f"Ferg d{fdeg}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("RQ1 | 03b: polynomial coefficients (BIC + Ferguson)\n" + "=" * 60)
    res = run()
    path = os.path.join(out_dir("RQ1"), "poly_coefficients.csv")
    res.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(res)} rows)")
