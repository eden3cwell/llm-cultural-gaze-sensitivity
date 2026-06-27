"""
15_JN_analysis.py

Full numeric Johnson-Neyman (floodlight) breakdown, IMPLICIT and EXPLICIT
framing combined into one CSV.

Output: outputs/summary/jn_analysis_full.csv
    (model, dv, framing, degree, treatment, reference, n_jn_points,
     jn_points, always_sig, never_sig, region_label)
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from config import (DATA_DIR, MODEL_FILES, MODEL_LABELS, ALL_MODELS,
                    ALPHA, out_dir)

warnings.filterwarnings("ignore")

DVS = {"engagement_score": "Engagement", "difficulty_delta": "Difficulty Delta"}

FORMULAS = {
    1: "{dv} ~ fixation_rate",
    2: "{dv} ~ fixation_rate + I(fixation_rate**2)",
    3: "{dv} ~ fixation_rate + I(fixation_rate**2) + I(fixation_rate**3)",
}

FERGUSON_THRESHOLD = 0.04
X_GRID = np.linspace(0, 100, 500)

IMPLICIT_PAIRS = [
    ("maori",    "baseline"),
    ("kaumatua", "baseline"),
    ("kaumatua", "maori"),
]
EXPLICIT_PAIRS = [
    ("gaze_aversion",        "baseline"),
    ("direct_gaze_explicit", "baseline"),
    ("gaze_aversion",        "direct_gaze_explicit"),
]


def _load_clean(model_key: str) -> pd.DataFrame:
    """All conditions, cleaned."""
    path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    df = df[df["parse_error"].isna() | (df["parse_error"].astype(str).isin(["", "nan", "None"]))]
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["engagement_score", "difficulty_delta", "fixation_rate", "condition"])
    df = df[(df.engagement_score >= 1) & (df.engagement_score <= 10)
            & (df.difficulty_delta >= -1) & (df.difficulty_delta <= 1)]
    return df


def ferguson_degree_for(dv: str, baseline: pd.DataFrame) -> int:
    """Two-condition practical screen (identical to 04_FergusonSelection.py),
    fit on baseline only."""
    fits = {}
    for deg, formula in FORMULAS.items():
        mod = smf.ols(formula.format(dv=dv), data=baseline).fit()
        fits[deg] = {"r2": round(mod.rsquared, 4)}

    base = None
    for deg in range(2, max(fits) + 1):
        if fits[deg]["r2"] - fits[1]["r2"] >= FERGUSON_THRESHOLD:
            base = deg
            break
    if base is None:
        return 1

    accepted = base
    for deg in range(base + 1, max(fits) + 1):
        if fits[deg]["r2"] - fits[accepted]["r2"] >= FERGUSON_THRESHOLD:
            accepted = deg
        else:
            break
    return accepted


def _region_label(grid: np.ndarray, sig_mask: np.ndarray) -> str:
    """Human-readable region description from a boolean significance mask."""
    if sig_mask.all():
        return "Significant across entire range [0, 100]"
    if not sig_mask.any():
        return "Not significant anywhere in [0, 100]"
    idx = np.where(sig_mask)[0]
    breaks = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, breaks + 1)
    spans = [f"{grid[s[0]]:.1f}-{grid[s[-1]]:.1f}%" for s in segments]
    return "Significant for fixation " + ", ".join(spans)


def get_effect_curve(df: pd.DataFrame, treatment_cond: str, reference_cond: str,
                     dv: str, degree: int, alpha: float = ALPHA):
    """Fit DV ~ cond_bin * poly(fixation, degree) and return (effect, se, jn)
    over X_GRID. effect[i] is the simple effect (treatment - reference) at
    grid point i; jn carries the significance mask, JN boundaries, and the
    human-readable region label."""
    subset = df[df["condition"].isin([reference_cond, treatment_cond])].copy()
    subset["cond_bin"] = (subset["condition"] == treatment_cond).astype(int)
    if subset["cond_bin"].nunique() < 2:
        return None

    poly_terms = " + ".join(
        ["fixation_rate"] + [f"I(fixation_rate**{j})" for j in range(2, degree + 1)]
    )
    formula = f"{dv} ~ cond_bin * ({poly_terms})"
    mod = smf.ols(formula, data=subset).fit()

    params, vcov = mod.params, mod.cov_params()
    t_crit = stats.t.ppf(1 - alpha / 2, int(mod.df_resid))

    eff_names = ["cond_bin", "cond_bin:fixation_rate"]
    eff_names += [f"cond_bin:I(fixation_rate ** {j})" for j in range(2, degree + 1)]
    eff_names = [n for n in eff_names if n in params.index]

    C = np.ones((len(X_GRID), len(eff_names)))
    for col, name in enumerate(eff_names):
        if name == "cond_bin":
            C[:, col] = 1.0
        elif name == "cond_bin:fixation_rate":
            C[:, col] = X_GRID
        else:
            j = int(name.split("**")[1].split(")")[0])
            C[:, col] = X_GRID ** j

    beta = params[eff_names].to_numpy()
    Sig  = vcov.loc[eff_names, eff_names].to_numpy()

    effect = C @ beta
    var    = np.maximum(np.einsum("ij,jk,ik->i", C, Sig, C), 1e-15)
    se     = np.sqrt(var)
    t_vals = effect / se

    sig_mask = np.abs(t_vals) >= t_crit
    flips = np.where(np.diff(sig_mask.astype(int)) != 0)[0]
    jn_points = sorted(float(X_GRID[i + 1]) for i in flips)

    jn = dict(sig_mask=sig_mask, jn_points=jn_points,
             always_sig=bool(sig_mask.all()), never_sig=bool(not sig_mask.any()),
             region_label=_region_label(X_GRID, sig_mask))
    return effect, se, jn


def run() -> pd.DataFrame:
    rows = []
    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            print(f"  [MISSING] {MODEL_FILES[model_key]}")
            continue
        data = _load_clean(model_key)
        label = MODEL_LABELS.get(model_key, model_key)
        baseline = data[data["condition"] == "baseline"]

        for dv, dv_label in DVS.items():
            k = ferguson_degree_for(dv, baseline)
            for framing, pairs in [("Implicit", IMPLICIT_PAIRS), ("Explicit", EXPLICIT_PAIRS)]:
                for treat, ref in pairs:
                    res = get_effect_curve(data, treat, ref, dv, k)
                    if res is None:
                        continue
                    _eff, _se, jn = res
                    rows.append({
                        "model": label, "dv": dv_label, "framing": framing,
                        "degree": k, "treatment": treat, "reference": ref,
                        "n_jn_points": len(jn["jn_points"]),
                        "jn_points": ";".join(f"{p:.1f}" for p in jn["jn_points"]),
                        "always_sig": jn["always_sig"], "never_sig": jn["never_sig"],
                        "region_label": jn["region_label"],
                    })
                    print(f"  {label:<12} {dv_label:<14} {framing:<9} "
                          f"{treat:<22} vs {ref:<22} k={k}  "
                          f"{jn['region_label'][:40]}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("15 | Full JN Analysis Breakdown (Implicit + Explicit)\n" + "=" * 60)
    results = run()
    path = os.path.join(out_dir("summary"), "jn_analysis_full.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")
