"""
RQ4 | Output 1: Structural point analysis — normalised polynomial curves.

For the difficulty_delta DV:
    1. Fits the RQ1 Ferguson-retained polynomial to each model x condition
       (falls back to local BIC only if the RQ1 CSV is unavailable)
    2. Normalises to [-1, +1] using baseline polynomial min/max as anchors
    3. Extracts structural points:
         r+       positive crossing (curve crosses zero upward)
         r_peak   maximum of normalised curve
         r-       negative crossing after peak (absent in monotone curves)
    4. Tests shifts in each structural point under treatment conditions
       vs baseline via one-sample t-test and Wilcoxon signed-rank
       (n=7, underpowered: report descriptively, not as confirmatory tests)

Outputs:
    outputs/new_experiment/RQ4/structural_points_baseline.csv
    outputs/new_experiment/RQ4/structural_point_shifts.csv
    outputs/new_experiment/RQ4/structural_point_tests.csv
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import (INFERENTIAL_MODELS, TREATMENT_CONDITIONS,
                    MODEL_LABELS, OUTPUT_DIR, out_dir)
from loaders import load_all, cell_means

warnings.filterwarnings("ignore")

X_RANGE = np.linspace(0, 100, 1000)


def load_rq1_degrees() -> dict:
    """Ferguson-retained degree per (model, dv) from RQ1, for consistency
    with RQ1-RQ3. Returns {} if the RQ1 CSV is unavailable, triggering the
    local-BIC fallback (with a warning)."""
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


def fit_poly(x, y, degree):
    return np.polyfit(x, y, degree)


def normalise(y_cond, y_bl_min, y_bl_max):
    span = y_bl_max - y_bl_min
    if span == 0:
        return np.zeros_like(y_cond)
    return 2 * (y_cond - y_bl_min) / span - 1


def find_crossings(x, y, level=0.0):
    """Return sorted list of x values where y crosses level."""
    roots = []
    for i in range(len(x) - 1):
        if (y[i] - level) * (y[i+1] - level) < 0:
            try:
                root = brentq(lambda t: np.interp(t, x, y) - level,
                              x[i], x[i+1])
                roots.append(float(root))
            except Exception:
                pass
    return sorted(roots)


def extract_structural_points(y_norm, x=X_RANGE):
    """Extract r+, r_peak, r- from normalised curve."""
    crossings = find_crossings(x, y_norm, level=0.0)
    peak_idx  = int(np.argmax(y_norm))
    r_peak    = float(x[peak_idx])

    # r+ = first upward zero crossing
    r_plus = None
    for c in crossings:
        idx = np.searchsorted(x, c)
        if idx < len(y_norm) - 1 and y_norm[min(idx + 1, len(y_norm)-1)] > y_norm[max(idx - 1, 0)]:
            r_plus = c
            break
    if r_plus is None and len(crossings) > 0:
        r_plus = crossings[0]

    # r- = first downward crossing after peak
    r_minus = None
    for c in crossings:
        if c > r_peak:
            idx = np.searchsorted(x, c)
            if idx < len(y_norm) - 1 and y_norm[min(idx + 1, len(y_norm)-1)] < y_norm[max(idx - 1, 0)]:
                r_minus = c
                break

    return r_plus, r_peak, r_minus


def run():
    dfs           = load_all()
    FERGUSON      = load_rq1_degrees()
    baseline_rows = []
    shift_rows    = []

    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        df_model = dfs[model_key]
        label    = MODEL_LABELS[model_key]
        dv       = "difficulty_delta"

        # Baseline polynomial at the Ferguson degree (fallback: local BIC)
        bl  = df_model[df_model["condition"] == "baseline"]
        cm  = cell_means(bl, dv)
        x_r = cm["fixation_rate"].values.astype(float)
        y_r = cm[f"{dv}_mean"].values
        deg = FERGUSON.get((label, dv), bic_degree(x_r, y_r))

        bl_coeffs   = fit_poly(x_r, y_r, deg)
        y_bl        = np.polyval(bl_coeffs, X_RANGE)
        y_bl_min    = float(y_bl.min())
        y_bl_max    = float(y_bl.max())

        rp, rpeak, rm = extract_structural_points(y_bl)
        baseline_rows.append({
            "model":   label,
            "degree":  deg,
            "r_plus":  round(rp,    2) if rp    is not None else np.nan,
            "r_peak":  round(rpeak, 2),
            "r_minus": round(rm,    2) if rm    is not None else np.nan,
            "bl_min":  round(y_bl_min, 4),
            "bl_max":  round(y_bl_max, 4),
        })

        # Treatment condition shifts (fitted at the same baseline degree)
        for cond in TREATMENT_CONDITIONS:
            tr = df_model[df_model["condition"] == cond]
            if len(tr) == 0:
                continue
            cm_c   = cell_means(tr, dv)
            y_c    = np.polyval(fit_poly(
                cm_c["fixation_rate"].values.astype(float),
                cm_c[f"{dv}_mean"].values, deg
            ), X_RANGE)
            cp, cpeak, cm_ = extract_structural_points(y_c)

            shift_rows.append({
                "model":          label,
                "condition":      cond,
                "delta_r_plus":   round((cp - rp),       2) if (cp is not None and rp    is not None) else np.nan,
                "delta_r_peak":   round((cpeak - rpeak),  2),
                "delta_r_minus":  round((cm_ - rm),       2) if (cm_ is not None and rm is not None) else np.nan,
            })
            print(f"  {label:<12} {cond:<22} "
                  f"Δr+={shift_rows[-1]['delta_r_plus']}  "
                  f"Δr_peak={shift_rows[-1]['delta_r_peak']}  "
                  f"Δr-={shift_rows[-1]['delta_r_minus']}")

    baseline_df = pd.DataFrame(baseline_rows)
    shift_df    = pd.DataFrame(shift_rows)

    # One-sample tests on shifts across models (n=7; underpowered, descriptive)
    test_rows = []
    for cond in TREATMENT_CONDITIONS:
        sub = shift_df[shift_df["condition"] == cond]
        for pt in ("delta_r_plus", "delta_r_peak", "delta_r_minus"):
            vals = sub[pt].dropna().values
            if len(vals) < 2:
                continue
            t, p_t = stats.ttest_1samp(vals, 0)
            w, p_w = (stats.wilcoxon(vals) if len(vals) >= 5
                      else (np.nan, np.nan))
            test_rows.append({
                "condition": cond,
                "point":     pt,
                "n":         len(vals),
                "mean_pp":   round(float(vals.mean()), 3),
                "sd":        round(float(vals.std(ddof=1)), 3),
                "t_stat":    round(float(t), 3),
                "p_t":       round(float(p_t), 4),
                "p_w":       round(float(p_w), 4) if not np.isnan(p_w) else np.nan,
            })
            print(f"  {cond:<22} {pt:<18} "
                  f"mean={vals.mean():+.2f}pp  t={t:+.3f}  p={p_t:.4f}")

    return baseline_df, shift_df, pd.DataFrame(test_rows)


if __name__ == "__main__":
    print("RQ4 | Output 1: Structural Points\n" + "=" * 60)
    bl_df, sh_df, te_df = run()

    for df, fname in [
        (bl_df, "structural_points_baseline.csv"),
        (sh_df, "structural_point_shifts.csv"),
        (te_df, "structural_point_tests.csv"),
    ]:
        path = os.path.join(out_dir("RQ4"), fname)
        df.to_csv(path, index=False)
        print(f"Saved: {path}  ({len(df)} rows)")