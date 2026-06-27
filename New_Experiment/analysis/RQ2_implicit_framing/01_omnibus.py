"""
RQ2 | Output 1: Implicit-framing omnibus (polynomial nested F-tests).

Establishes whether the implicit identity labels (maori, kaumatua) shift the
baseline response surface, and quantifies the shift with effect sizes so that
near-deterministic (large-n) significance is not over-interpreted. Partial
eta^2 is the primary cross-model comparison metric; p-values are retained for
reference only and are inert under the near-deterministic output.

Design rationale:
  - Restricted to baseline + maori + kaumatua (implicit only); the explicit gaze
    conditions are excluded so the test reflects implicit framing rather than
    being carried by the much larger explicit-condition effects.
  - Fixation modelled at the RQ1 FERGUSON-RETAINED polynomial degree per
    (model, DV):
        DV ~ condition * poly(fixation_rate, k),
    so condition effects are estimated against the same functional form used in
    RQ1. The contrast therefore isolates positional and within-form shape
    changes; whether a label alters the response BEYOND the baseline form is
    captured by the condition x fixation interaction term (eta2_interaction),
    which is reported separately.
  - Significance via nested OLS F-tests (a mixed-model LRT is not used: the
    run_id random-effect variance is ~0 / ICC ~ 0, so the mixed model is
    unnecessary and fails to converge):
        omnibus  (any condition effect):  poly(fix)             vs  condition*poly(fix)
        interaction (moderation only):    condition + poly(fix) vs  condition*poly(fix)
    Per-label nested F (baseline vs each label) and the within-culture contrast
    (kaumatua vs maori) are also reported.
  - Effect sizes: partial eta^2 = (RSS_r - RSS_f) / RSS_r, and dR2 (incremental
    variance over the reduced model). NO multiple-comparison correction is
    applied (effect-size-led, consistent with RQ1).

Output: outputs/new_experiment/RQ2/omnibus_results.csv
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import (INFERENTIAL_MODELS, IMPLICIT_CONDITIONS, MODEL_LABELS,
                    ALPHA, OUTPUT_DIR, out_dir)
from loaders import load_all

warnings.filterwarnings("ignore")

DVS = ["engagement_score", "difficulty_delta"]

# Fallback RQ1 Ferguson-retained degrees, used only if the RQ1 CSV is missing.
# Engagement: all degree 1 except Qwen-14B (2).
# Difficulty: all degree 2 except Llama-11B and Qwen-7B (3).
FALLBACK_DEG = {
    ("Llama-3B",  "engagement_score"): 1, ("Llama-3B",  "difficulty_delta"): 2,
    ("Llama-8B",  "engagement_score"): 1, ("Llama-8B",  "difficulty_delta"): 2,
    ("Llama-11B", "engagement_score"): 1, ("Llama-11B", "difficulty_delta"): 3,
    ("Mistral-7B","engagement_score"): 1, ("Mistral-7B","difficulty_delta"): 2,
    ("Qwen-7B",   "engagement_score"): 1, ("Qwen-7B",   "difficulty_delta"): 3,
    ("Qwen-14B",  "engagement_score"): 2, ("Qwen-14B",  "difficulty_delta"): 2,
    ("QwenVL-7B", "engagement_score"): 1, ("QwenVL-7B", "difficulty_delta"): 2,
}


def load_rq1_degrees() -> dict:
    """Read RQ1 degrees keyed by (model_label, dv). Prefers the Ferguson-
    retained degree; falls back to selected_degree for older CSVs."""
    path = os.path.join(OUTPUT_DIR, "RQ1", "bic_polynomial_results.csv")
    if not os.path.exists(path):
        print(f"  [info] RQ1 degrees not found at {path}; using fallback.")
        return dict(FALLBACK_DEG)
    rq1 = pd.read_csv(path)
    dv_map = {"Engagement Score": "engagement_score",
              "Difficulty Delta": "difficulty_delta"}
    deg_col = ("ferguson_degree" if "ferguson_degree" in rq1.columns
               else "selected_degree")
    deg = {}
    for _, r in rq1.iterrows():
        dv = dv_map.get(str(r["dv"]).strip(), str(r["dv"]).strip())
        deg[(str(r["model"]).strip(), dv)] = int(r[deg_col])
    return deg


def _rss(X: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares of the OLS fit of y on X."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid)


def _poly(x: np.ndarray, k: int) -> np.ndarray:
    """Columns x, x^2, ..., x^k (x already scaled to [0, 1])."""
    return np.column_stack([x ** j for j in range(1, k + 1)])


def _compare(rss_r: float, rss_f: float, p_r: int, p_f: int,
             n: int, tss: float) -> dict:
    """Nested F-test + effect sizes comparing reduced to full OLS model."""
    df1 = p_f - p_r
    df2 = n - p_f
    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    p = float(scipy_stats.f.sf(F, df1, df2))
    eta2 = (rss_r - rss_f) / rss_r                 # partial eta-squared
    dR2 = (rss_r - rss_f) / tss                    # incremental R^2 over reduced
    return {"F": round(F, 3), "df1": df1, "df2": df2, "p": round(p, 5),
            "eta2": round(eta2, 4), "dR2": round(dR2, 4)}


def _design(df: pd.DataFrame, dv: str, k: int, conds: list[str]):
    """Return reduced (poly), mid (poly+cond), full (poly*cond) matrices, y, n, tss."""
    d = df[df["condition"].isin(conds)].copy()
    x = d["fixation_rate"].to_numpy(float) / 100.0          # scale to [0, 1]
    y = d[dv].to_numpy(float)
    n = len(y)
    tss = float(((y - y.mean()) ** 2).sum())
    ones = np.ones((n, 1))
    P = _poly(x, k)
    dums = np.column_stack([(d["condition"] == c).to_numpy(float)
                            for c in conds[1:]])             # first cond = reference
    inter = np.column_stack([dums[:, [j]] * P for j in range(dums.shape[1])])
    X_poly = np.hstack([ones, P])                            # reduced: poly only
    X_mid  = np.hstack([ones, P, dums])                      # + condition main effects
    X_full = np.hstack([ones, P, dums, inter])               # + interaction
    return X_poly, X_mid, X_full, y, n, tss


def run_omnibus(df: pd.DataFrame, dv: str, k: int) -> dict:
    """Omnibus, interaction, per-label, and within-culture tests for one model/DV."""
    conds = ["baseline"] + list(IMPLICIT_CONDITIONS)         # baseline, maori, kaumatua
    Xp, Xm, Xf, y, n, tss = _design(df, dv, k, conds)
    rss_p, rss_m, rss_f = _rss(Xp, y), _rss(Xm, y), _rss(Xf, y)

    omni  = _compare(rss_p, rss_f, Xp.shape[1], Xf.shape[1], n, tss)  # any condition effect
    inter = _compare(rss_m, rss_f, Xm.shape[1], Xf.shape[1], n, tss)  # moderation only

    out = {
        "degree": k, "n": n, "r2_full": round(1 - rss_f / tss, 4),
        "F_condition": omni["F"], "df_condition": f"{omni['df1']},{omni['df2']}",
        "p_condition": omni["p"], "eta2_condition": omni["eta2"],
        "dR2_condition": omni["dR2"], "sig_condition": bool(omni["p"] < ALPHA),
        "F_interaction": inter["F"], "df_interaction": f"{inter['df1']},{inter['df2']}",
        "p_interaction": inter["p"], "eta2_interaction": inter["eta2"],
    }

    # Per-label: baseline vs each implicit condition separately
    for cond in IMPLICIT_CONDITIONS:
        Xp2, _Xm2, Xf2, y2, n2, tss2 = _design(df, dv, k, ["baseline", cond])
        res = _compare(_rss(Xp2, y2), _rss(Xf2, y2),
                       Xp2.shape[1], Xf2.shape[1], n2, tss2)
        out[f"F_{cond}"]    = res["F"]
        out[f"p_{cond}"]    = res["p"]
        out[f"eta2_{cond}"] = res["eta2"]
        out[f"dR2_{cond}"]  = res["dR2"]

    # Within-culture contrast: kaumatua vs maori (no baseline)
    Xp3, _Xm3, Xf3, y3, n3, tss3 = _design(df, dv, k, list(IMPLICIT_CONDITIONS))
    kvm = _compare(_rss(Xp3, y3), _rss(Xf3, y3),
                   Xp3.shape[1], Xf3.shape[1], n3, tss3)
    out["F_kaum_vs_maori"]    = kvm["F"]
    out["df_kaum_vs_maori"]   = f"{kvm['df1']},{kvm['df2']}"
    out["p_kaum_vs_maori"]    = kvm["p"]
    out["eta2_kaum_vs_maori"] = kvm["eta2"]
    return out


def run() -> pd.DataFrame:
    dfs = load_all()
    deg = load_rq1_degrees()
    rows = []

    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        label = MODEL_LABELS[model_key]
        for dv in DVS:
            k = deg.get((label, dv), FALLBACK_DEG[(label, dv)])
            result = run_omnibus(dfs[model_key], dv, k)
            rows.append({"model": label, "dv": dv, **result})
            print(f"  {label:<11} {dv:<18} k={result['degree']}  "
                  f"eta2_cond={result['eta2_condition']:<7} "
                  f"M={result['eta2_maori']:<7} K={result['eta2_kaumatua']:<7} "
                  f"int={result['eta2_interaction']:<7} "
                  f"KvM eta2={result['eta2_kaum_vs_maori']}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("RQ2 | Output 1: Implicit-framing omnibus (polynomial nested F)\n" + "=" * 62)
    results = run()
    path = os.path.join(out_dir("RQ2"), "omnibus_results.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")