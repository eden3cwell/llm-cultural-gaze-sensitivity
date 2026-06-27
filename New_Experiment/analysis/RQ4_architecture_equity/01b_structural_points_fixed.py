"""
01b_structural_points_fixed.py  —  BUG-FIX GENERATOR (additive; 01 untouched)

01_structural_points.py finds crossings on the NORMALISED curve, where 0 is the
raw-delta min-max midpoint, not delta = 0. Its r+ / r- therefore do not mean
"the difficulty curve crosses zero", which is what the thesis text and the main
table `tab:rq4_structural` report. (The main table happens to hold the correct
delta=0 values; the committed script reproduces the WRONG appendix values in
`tab:app_struct_base`.)

This script computes the structural points the way the thesis defines them:
    r+      first UPWARD crossing of delta = 0   (task starts being made harder)
    r_peak  fixation at maximum fitted delta
    r-      first DOWNWARD crossing of delta = 0 after the peak
all on the RAW (un-normalised) baseline delta curve, fitted at the RQ1
Ferguson-retained degree. Reproduces the main table exactly
(e.g. Llama-3B r+ = 26.5, Mistral r- = 89.7).

Self-scoped to the seven inferential models (unaffected by the 32B drift).

Output: outputs/new_experiment/RQ4/structural_points_fixed.csv
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
# RQ1 Ferguson-retained degrees for the difficulty_delta DV.
FERGUSON_DELTA = {"Llama-3B": 2, "Llama-8B": 2, "Llama-11B": 3, "Mistral-7B": 2,
                  "Qwen-7B": 3, "Qwen-14B": 2, "QwenVL-7B": 2}
X = np.linspace(0, 100, 2000)


def _clean(df):
    df = df[df["parse_error"].isna() | (df["parse_error"].astype(str).isin(["", "nan", "None"]))].copy()
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["difficulty_delta", "fixation_rate", "condition"])
    return df[(df.difficulty_delta >= -1) & (df.difficulty_delta <= 1)]


def _zero_crossings(y, level=0.0):
    """Return list of (x, direction) where y crosses `level`; +1 upward."""
    out = []
    for i in range(len(y) - 1):
        if (y[i] - level) * (y[i + 1] - level) < 0:
            xc = X[i] + (level - y[i]) * (X[i + 1] - X[i]) / (y[i + 1] - y[i])
            out.append((float(xc), 1 if y[i + 1] > y[i] else -1))
    return out


def _structural(y):
    cross = _zero_crossings(y)
    peak = float(X[int(np.argmax(y))])
    r_plus = next((x for x, d in cross if d > 0), None)
    if r_plus is None and cross:
        r_plus = cross[0][0]
    r_minus = next((x for x, d in cross if d < 0 and x > peak), None)
    return r_plus, peak, r_minus


def run():
    rows = []
    for stem, label in ROSTER:
        df = _clean(pd.read_csv(os.path.join(DATA_DIR, f"{stem}.csv")))
        b = df[df.condition == "baseline"]
        cm = b.groupby("fixation_rate")["difficulty_delta"].mean().reset_index()
        x = cm.fixation_rate.values.astype(float)
        y = cm.difficulty_delta.values
        deg = FERGUSON_DELTA[label]
        yfit = np.polyval(np.polyfit(x, y, deg), X)
        rp, pk, rm = _structural(yfit)
        rows.append({
            "model": label, "degree": deg,
            "r_plus": None if rp is None else round(rp, 1),
            "r_peak": round(pk, 1),
            "r_minus": None if rm is None else round(rm, 1),
            "delta_min": round(float(yfit.min()), 4),
            "delta_max": round(float(yfit.max()), 4),
        })
        print(f"  {label:<11} deg {deg}  r+={rows[-1]['r_plus']}  "
              f"r_peak={rows[-1]['r_peak']}  r-={rows[-1]['r_minus']}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("RQ4 | 01b: structural points (delta=0 crossings, FIXED)\n" + "=" * 60)
    res = run()
    path = os.path.join(out_dir("RQ4"), "structural_points_fixed.csv")
    res.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(res)} rows)")
