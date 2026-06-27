"""
Robustness Check | Output 1: Prompt robustness analysis.

Tests whether the fixation-engagement response curves are functionally
equivalent across three prompt variants for the baseline condition.

Variant sources:
    standard   — main model CSV (Qwen2.5-14B.csv / Llama-3.2-3B.csv),
                 condition == "baseline"
    variant_a  — results/new_experiment/robustness_check.csv
    variant_b  — results/new_experiment/robustness_check.csv

Equivalence criteria (two must both be met):
    sMAD < 0.20  — standardised mean absolute deviation between fitted curves,
                   thresholded at Cohen's (1988) small-effect boundary.
                   Standardised by the baseline SD of the DV so the threshold
                   is scale-invariant and comparable across DVs.
    ICC  > 0.75  — intraclass correlation coefficient ICC(3,1) of the three
                   fitted curves evaluated on a common 0–100 grid,
                   thresholded at Koo & Mae's (2016) "good" reliability cut-off.

Secondary diagnostic (reported but not used for pass/fail):
    β₁ sign consistency — whether the linear slope is positive across all
                          variants.  Addresses the directional robustness of
                          the cognitive-overload interpretation.

References:
    Cohen, J. (1988). Statistical power analysis for the behavioral sciences
        (2nd ed.). Lawrence Erlbaum.
    Koo, T. K., & Mae, M. Y. (2016). A guideline of selecting and reporting
        intraclass correlation coefficients for reliability research.
        Journal of Chiropractic Medicine, 15(2), 155–163.

Outputs:
    outputs/new_experiment/robustness/robustness_polynomial_results.csv
    outputs/new_experiment/robustness/robustness_summary.csv
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import MODEL_LABELS, out_dir
from loaders import load_model, load_robustness

warnings.filterwarnings("ignore")

ROBUSTNESS_MODEL_MAP = {
    "Qwen2.5-14B": "Qwen/Qwen2.5-14B-Instruct",
    "Llama-3.2-3B": "meta-llama/Llama-3.2-3B-Instruct",
}

VARIANTS = ["standard", "variant_a", "variant_b"]
DVS      = ["engagement_score", "difficulty_delta"]

# Evaluation grid for curve comparison
X_GRID = np.linspace(0, 100, 101)

# Equivalence thresholds (academically grounded — see module docstring)
SMAD_THRESHOLD = 0.20   # Cohen (1988) small-effect boundary
ICC_THRESHOLD  = 0.75   # Koo & Mae (2016) "good" reliability


# ── Statistical helpers ────────────────────────────────────────────────────────

def bic_degree(x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> int:
    n = len(x)
    best_bic, best_deg = np.inf, 1
    for deg in range(1, max_degree + 1):
        c   = np.polyfit(x, y, deg)
        sse = float(np.sum((y - np.polyval(c, x)) ** 2))
        bic = n * np.log(sse / n) + (deg + 1) * np.log(n)
        if bic < best_bic:
            best_bic, best_deg = bic, deg
    return best_deg


def slope_inference(df: pd.DataFrame, dv: str, degree: int) -> dict:
    x = df["fixation_rate"].values.astype(float)
    y = df[dv].values

    X_cols = {"fixation_rate": x}
    if degree >= 2:
        X_cols["fixation_rate_sq"] = x ** 2
    if degree >= 3:
        X_cols["fixation_rate_cu"] = x ** 3

    X   = sm.add_constant(pd.DataFrame(X_cols))
    mod = sm.OLS(y, X).fit()

    beta1   = float(mod.params.get("fixation_rate", np.nan))
    beta1_p = float(mod.pvalues.get("fixation_rate", np.nan))

    return {
        "beta1":      beta1,
        "beta1_p":    beta1_p,
        "r2":         float(mod.rsquared),
        "beta1_sign": "positive" if beta1 > 0 else "negative",
    }


def icc_consistency(mat: np.ndarray) -> float:
    """
    ICC(3,1): two-way mixed effects, single measures, consistency.

    mat shape: (n_raters, n_subjects) — here (n_variants, len(X_GRID)).
    Treats x-grid points as subjects and prompt variants as raters.

    Koo & Mae (2016): ICC > 0.75 = good, > 0.90 = excellent.
    """
    k, n = mat.shape
    if k < 2 or n < 2:
        return np.nan
    grand_mean = mat.mean()
    ss_rows    = n * np.sum((mat.mean(axis=1) - grand_mean) ** 2)
    ss_cols    = k * np.sum((mat.mean(axis=0) - grand_mean) ** 2)
    ss_total   = np.sum((mat - grand_mean) ** 2)
    ss_error   = ss_total - ss_rows - ss_cols
    ms_cols    = ss_cols  / (n - 1)
    ms_error   = ss_error / ((k - 1) * (n - 1))
    denom      = ms_cols + (k - 1) * ms_error
    return float((ms_cols - ms_error) / denom) if denom != 0 else np.nan


def curve_divergence(
    variants: dict[str, pd.DataFrame],
    dv: str,
    baseline_sd: float,
) -> dict:
    """
    Fit BIC-selected polynomial for each variant, evaluate on X_GRID,
    and quantify how much the curves diverge.

    Returns
    -------
    mean_mad   : mean |deviation| across variants in raw DV units
    max_mad    : worst-case |deviation| — where curve spread is highest
    mean_sMAD  : mean_mad / baseline_sd  (Cohen 1988 standardised units)
    max_sMAD   : max_mad  / baseline_sd
    icc        : ICC(3,1) of fitted curves (Koo & Mae 2016)
    sign_consistent : all variants agree on direction of β₁
    equivalent : mean_sMAD < 0.20 AND icc > 0.75
    """
    fitted_curves = {}
    signs         = []

    for variant, df_v in variants.items():
        if len(df_v) < 10:
            continue
        cm    = df_v.groupby("fixation_rate")[dv].mean().reset_index()
        x_r   = cm["fixation_rate"].values.astype(float)
        y_r   = cm[dv].values
        deg   = bic_degree(x_r, y_r)
        coeff = np.polyfit(x_r, y_r, deg)
        fitted_curves[variant] = np.polyval(coeff, X_GRID)

        inf = slope_inference(df_v, dv, deg)
        signs.append(inf["beta1_sign"])

    nan_result = {k: np.nan for k in
                  ["mean_mad", "max_mad", "mean_sMAD", "max_sMAD", "icc"]}
    nan_result.update({"sign_consistent": False, "equivalent": False})

    if len(fitted_curves) < 2:
        return nan_result

    mat = np.stack(list(fitted_curves.values()), axis=0)   # (n_variants, 101)

    # MAD: for each x, average |deviation from the cross-variant mean|
    mad_per_x = np.mean(np.abs(mat - mat.mean(axis=0, keepdims=True)), axis=0)
    mean_mad  = float(mad_per_x.mean())
    max_mad   = float(mad_per_x.max())

    # Standardise by baseline SD (Cohen 1988)
    if baseline_sd > 0:
        mean_smad = mean_mad / baseline_sd
        max_smad  = max_mad  / baseline_sd
    else:
        mean_smad = max_smad = np.nan

    icc             = icc_consistency(mat)
    sign_consistent = len(set(signs)) == 1

    equivalent = (
        (not np.isnan(mean_smad) and mean_smad < SMAD_THRESHOLD) and
        (not np.isnan(icc)       and icc        > ICC_THRESHOLD)
    )

    return {
        "mean_mad":        round(mean_mad,  4),
        "max_mad":         round(max_mad,   4),
        "mean_sMAD":       round(mean_smad, 4) if not np.isnan(mean_smad) else np.nan,
        "max_sMAD":        round(max_smad,  4) if not np.isnan(max_smad)  else np.nan,
        "icc":             round(icc,       4) if not np.isnan(icc)        else np.nan,
        "sign_consistent": sign_consistent,
        "equivalent":      equivalent,
    }


# ── Data loading ───────────────────────────────────────────────────────────────

def load_model_variants(
    model_key: str,
    df_rob: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    full_id = ROBUSTNESS_MODEL_MAP[model_key]
    result  = {}

    df_main = load_model(model_key)
    if df_main is not None:
        std = df_main[df_main["condition"] == "baseline"].copy()
        if len(std) > 0:
            result["standard"] = std

    if df_rob is not None and len(df_rob) > 0:
        for variant in ("variant_a", "variant_b"):
            sub = df_rob[
                (df_rob["model"] == full_id) &
                (df_rob["prompt_variant"] == variant)
            ].copy()
            if len(sub) > 0:
                result[variant] = sub

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_rob    = load_robustness()
    poly_rows = []
    summ_rows = []

    for model_key in ROBUSTNESS_MODEL_MAP:
        label    = MODEL_LABELS.get(model_key, model_key)
        variants = load_model_variants(model_key, df_rob)

        if not variants:
            print(f"  [SKIP] {label}: no data found")
            continue

        found   = list(variants.keys())
        missing = [v for v in VARIANTS if v not in found]
        if missing:
            print(f"  [WARN] {label}: missing variants {missing}")

        print(f"\n  {label}  (variants: {found})")

        for dv in DVS:
            # Baseline SD from standard variant for standardisation
            std_df      = variants.get("standard")
            baseline_sd = float(std_df[dv].std(ddof=1)) if std_df is not None else np.nan

            # Per-variant polynomial stats
            for variant in VARIANTS:
                sub = variants.get(variant)
                if sub is None or len(sub) < 10:
                    if sub is not None:
                        print(f"    [WARN] {label}/{variant}/{dv}: n={len(sub)} < 10")
                    continue

                cm  = sub.groupby("fixation_rate")[dv].mean().reset_index()
                x_r = cm["fixation_rate"].values.astype(float)
                y_r = cm[dv].values
                deg = bic_degree(x_r, y_r)
                inf = slope_inference(sub, dv, deg)

                poly_rows.append({
                    "model":          label,
                    "prompt_variant": variant,
                    "dv":             dv,
                    "n_obs":          len(sub),
                    "bic_degree":     deg,
                    "beta1":          round(inf["beta1"],   6),
                    "beta1_p":        round(inf["beta1_p"], 4),
                    "beta1_sign":     inf["beta1_sign"],
                    "r2":             round(inf["r2"],      4),
                })
                print(f"    {variant:<12} {dv:<22} "
                      f"deg={deg}  β₁={inf['beta1']:+.4f}  "
                      f"p={inf['beta1_p']:.4f}  R²={inf['r2']:.3f}")

            # Curve-divergence equivalence test
            div = curve_divergence(variants, dv, baseline_sd)

            verdict = "EQUIVALENT" if div["equivalent"] else "DIVERGENT"
            smad_flag = (f"sMAD={div['mean_sMAD']:.3f}"
                         f"{'✓' if not np.isnan(div['mean_sMAD']) and div['mean_sMAD'] < SMAD_THRESHOLD else '✗'}"
                         f"(<{SMAD_THRESHOLD})")
            icc_flag  = (f"ICC={div['icc']:.3f}"
                         f"{'✓' if not np.isnan(div['icc']) and div['icc'] > ICC_THRESHOLD else '✗'}"
                         f"(>{ICC_THRESHOLD})")
            sign_flag = "sign=consistent" if div["sign_consistent"] else "sign=INCONSISTENT"
            print(f"    --> {dv:<22} {verdict}  {smad_flag}  {icc_flag}  {sign_flag}")

            summ_rows.append({
                "model":        label,
                "dv":           dv,
                "n_variants":   len(found),
                "baseline_sd":  round(baseline_sd, 4) if not np.isnan(baseline_sd) else np.nan,
                **div,
                "smad_thresh":  SMAD_THRESHOLD,
                "icc_thresh":   ICC_THRESHOLD,
            })

    return pd.DataFrame(poly_rows), pd.DataFrame(summ_rows)


if __name__ == "__main__":
    print("Robustness Check | Output 1: Curve-Divergence Equivalence\n" + "=" * 60)
    print(f"  Thresholds: sMAD < {SMAD_THRESHOLD} (Cohen 1988)  |  "
          f"ICC > {ICC_THRESHOLD} (Koo & Mae 2016)\n")
    poly_df, summ_df = run()

    if len(poly_df) > 0:
        p1 = os.path.join(out_dir("robustness"), "robustness_polynomial_results.csv")
        p2 = os.path.join(out_dir("robustness"), "robustness_summary.csv")
        poly_df.to_csv(p1, index=False)
        summ_df.to_csv(p2, index=False)
        print(f"\nSaved: {p1}  ({len(poly_df)} rows)")
        print(f"Saved: {p2}  ({len(summ_df)} rows)")
    else:
        print("\nNo output — check main model CSVs exist and "
              "robustness_check.csv has been generated by slurm/robustness_check.sh")
