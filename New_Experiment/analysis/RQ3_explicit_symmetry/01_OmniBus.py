"""
RQ3 | Output 1: Explicit-framing omnibus (polynomial nested F-tests).

Mirrors the RQ2 omnibus exactly, with the explicit conditions
(gaze_aversion = WEIRD-contradicting; direct_gaze_explicit = WEIRD-affirming)
in place of the implicit labels.

Answers two RQ3 sub-questions directly from effect sizes:
  - Override: eta2 of gaze_aversion vs baseline (does the contradicting cue
    move the heuristic).
  - Ceiling: eta2 of direct_gaze_explicit vs baseline (a small value indicates
    the affirming cue cannot amplify an already-WEIRD default).
  - Symmetry: the GA-vs-DGE contrast, and the disparity between eta2_GA and
    eta2_DGE, quantify directional (a)symmetry without a sum-of-d index.

Conditions: baseline + gaze_aversion + direct_gaze_explicit.
Fixation modelled at the RQ1 Ferguson-retained degree per (model, DV).
No multiple-comparison correction (effect-size-led, consistent with RQ1/RQ2).

Output: outputs/new_experiment/RQ3/omnibus_results.csv
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import (INFERENTIAL_MODELS, EXPLICIT_CONDITIONS, MODEL_LABELS,
                    ALPHA, OUTPUT_DIR, out_dir)
from loaders import load_all

warnings.filterwarnings("ignore")

DVS = ["engagement_score", "difficulty_delta"]

# Fallback RQ1 Ferguson degrees (used only if the RQ1 CSV is missing).
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
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid)


def _poly(x: np.ndarray, k: int) -> np.ndarray:
    return np.column_stack([x ** j for j in range(1, k + 1)])


def _compare(rss_r: float, rss_f: float, p_r: int, p_f: int,
             n: int, tss: float) -> dict:
    df1 = p_f - p_r
    df2 = n - p_f
    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    p = float(scipy_stats.f.sf(F, df1, df2))
    eta2 = (rss_r - rss_f) / rss_r
    dR2 = (rss_r - rss_f) / tss
    return {"F": round(F, 3), "df1": df1, "df2": df2, "p": round(p, 5),
            "eta2": round(eta2, 4), "dR2": round(dR2, 4)}


def _design(df: pd.DataFrame, dv: str, k: int, conds: list[str]):
    d = df[df["condition"].isin(conds)].copy()
    x = d["fixation_rate"].to_numpy(float) / 100.0
    y = d[dv].to_numpy(float)
    n = len(y)
    tss = float(((y - y.mean()) ** 2).sum())
    ones = np.ones((n, 1))
    P = _poly(x, k)
    dums = np.column_stack([(d["condition"] == c).to_numpy(float)
                            for c in conds[1:]])
    inter = np.column_stack([dums[:, [j]] * P for j in range(dums.shape[1])])
    X_poly = np.hstack([ones, P])
    X_mid  = np.hstack([ones, P, dums])
    X_full = np.hstack([ones, P, dums, inter])
    return X_poly, X_mid, X_full, y, n, tss


def run_omnibus(df: pd.DataFrame, dv: str, k: int) -> dict:
    conds = ["baseline"] + list(EXPLICIT_CONDITIONS)   # baseline, GA, DGE
    Xp, Xm, Xf, y, n, tss = _design(df, dv, k, conds)
    rss_p, rss_m, rss_f = _rss(Xp, y), _rss(Xm, y), _rss(Xf, y)

    omni  = _compare(rss_p, rss_f, Xp.shape[1], Xf.shape[1], n, tss)
    inter = _compare(rss_m, rss_f, Xm.shape[1], Xf.shape[1], n, tss)

    out = {
        "degree": k, "n": n, "r2_full": round(1 - rss_f / tss, 4),
        "F_condition": omni["F"], "df_condition": f"{omni['df1']},{omni['df2']}",
        "p_condition": omni["p"], "eta2_condition": omni["eta2"],
        "dR2_condition": omni["dR2"], "sig_condition": bool(omni["p"] < ALPHA),
        "F_interaction": inter["F"], "df_interaction": f"{inter['df1']},{inter['df2']}",
        "p_interaction": inter["p"], "eta2_interaction": inter["eta2"],
    }

    # Per-cue: baseline vs each explicit condition separately
    for cond in EXPLICIT_CONDITIONS:
        Xp2, _Xm2, Xf2, y2, n2, tss2 = _design(df, dv, k, ["baseline", cond])
        res = _compare(_rss(Xp2, y2), _rss(Xf2, y2),
                       Xp2.shape[1], Xf2.shape[1], n2, tss2)
        out[f"F_{cond}"]    = res["F"]
        out[f"p_{cond}"]    = res["p"]
        out[f"eta2_{cond}"] = res["eta2"]
        out[f"dR2_{cond}"]  = res["dR2"]

    # Between-cue contrast: gaze_aversion vs direct_gaze_explicit (the symmetry test)
    Xp3, _Xm3, Xf3, y3, n3, tss3 = _design(df, dv, k, list(EXPLICIT_CONDITIONS))
    gvd = _compare(_rss(Xp3, y3), _rss(Xf3, y3),
                   Xp3.shape[1], Xf3.shape[1], n3, tss3)
    out["F_ga_vs_dge"]    = gvd["F"]
    out["df_ga_vs_dge"]   = f"{gvd['df1']},{gvd['df2']}"
    out["p_ga_vs_dge"]    = gvd["p"]
    out["eta2_ga_vs_dge"] = gvd["eta2"]
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
                  f"GA={result['eta2_gaze_aversion']:<7} "
                  f"DGE={result['eta2_direct_gaze_explicit']:<7} "
                  f"int={result['eta2_interaction']:<7} "
                  f"GvD eta2={result['eta2_ga_vs_dge']}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("RQ3 | Output 1: Explicit-framing omnibus (polynomial nested F)\n" + "=" * 62)
    results = run()
    path = os.path.join(out_dir("RQ3"), "omnibus_results.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")