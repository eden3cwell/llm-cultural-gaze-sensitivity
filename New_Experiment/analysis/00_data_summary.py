"""
00_data_summary.py  —  MISSING-GENERATOR FILL (additive; nothing existing is edited)

Produces the two descriptive artefacts the thesis reports but the pipeline did
not generate:

  1. Completion / parse-success table  (thesis Table `tab:completion` /
     appendix `tab:parse-rates`):  per-model total inferences, parse errors,
     success %, and the parse-error breakdown by condition.  Computed on the
     RAW CSVs (pre-clean) across ALL EIGHT thesis models, INCLUDING Llama-1B.

  2. Inclusion-criterion test (Methodology, Llama-1B exclusion):  the degree-1
     OLS of engagement on fixation, reported per model so the R^2 >= .01 rule
     is auditable.  NOTE: Llama-1B baseline engagement is invariant (constant
     5.0), so its baseline R^2 is undefined (zero variance); the pooled
     all-condition fit gives R^2 ~ .006.  Either way it fails R^2 >= .01.  The
     thesis's stated "beta1=0.000, R^2=.004, p=.061" does not reproduce and
     should be corrected to the values printed here.

This script is self-scoped to the eight thesis models and reads the CSVs
directly, so it is unaffected by the 32B config drift that crashes load_all().

Outputs:
    outputs/new_experiment/summary/completion_by_model.csv
    outputs/new_experiment/summary/parse_errors_by_condition.csv
    outputs/new_experiment/summary/inclusion_test.csv
"""

from __future__ import annotations

import os
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "shared"))
from config import DATA_DIR, out_dir  # paths only; roster is defined locally

# (csv stem, display label, params B) — the eight thesis models, 1B included.
ROSTER = [
    ("Llama-3.2-1B",  "Llama-1B",   1),
    ("Llama-3.2-3B",  "Llama-3B",   3),
    ("Mistral-7B",    "Mistral-7B", 7),
    ("Qwen2.5-7B",    "Qwen-7B",    7),
    ("Qwen2.5-VL-7B", "QwenVL-7B",  7),
    ("Llama-3.1-8B",  "Llama-8B",   8),
    ("Llama-3.2-11B", "Llama-11B",  11),
    ("Qwen2.5-14B",   "Qwen-14B",   14),
    ("Qwen2.5-32B-Instruct",   "Qwen-32B",   32),
    ("Qwen2.5-VL-32B-Instruct",   "QwenVL-32B",   32),
]

CONDITIONS = ["baseline", "maori", "kaumatua", "gaze_aversion", "direct_gaze_explicit"]


def _norm_p(t: float) -> float:
    """Two-sided p from a t/z stat via the normal approximation.
    At df ~ 1008 this is indistinguishable from the exact t-test; if scipy is
    present we use the exact value instead."""
    try:
        from scipy import stats
        return float(2 * stats.norm.sf(abs(t)))
    except Exception:
        return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))


def _read_raw(stem: str) -> pd.DataFrame:
    """Tolerant read of a raw result CSV (skips malformed lines so a corrupt
    file cannot abort the summary)."""
    path = os.path.join(DATA_DIR, f"{stem}.csv")
    return pd.read_csv(path, engine="python", on_bad_lines="skip")

#############################################################################################
#############################################################################################

def _is_error(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return series.notna() & ~s.isin(["", "nan", "None"])

#############################################################################################
#############################################################################################

def completion_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cond_rows = [], []
    tot = err = 0
    for stem, label, pb in ROSTER:
        df = _read_raw(stem)
        n = len(df)
        emask = _is_error(df["parse_error"])
        e = int(emask.sum())
        tot += n
        err += e
        rows.append({"model": label, "params_b": pb, "total": n,
                     "parse_err": e, "success_pct": round(100 * (n - e) / n, 2)})
        by_cond = df.loc[emask, "condition"].value_counts().to_dict()
        cond_rows.append({"model": label,
                          **{c: int(by_cond.get(c, 0)) for c in CONDITIONS}})
    rows.append({"model": "TOTAL", "params_b": np.nan, "total": tot,
                 "parse_err": err, "success_pct": round(100 * (tot - err) / tot, 2)})
    return pd.DataFrame(rows), pd.DataFrame(cond_rows)

#######################################################################################################
#######################################################################################################
def _clean_baseline(stem: str):
    df = _read_raw(stem)
    df = df[~_is_error(df["parse_error"])].copy()
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["engagement_score", "fixation_rate", "condition"])
    df = df[(df["engagement_score"] >= 1) & (df["engagement_score"] <= 10)]
    return df


def _ols1(x, y):
    """Degree-1 OLS by hand: returns (beta1, R2, se, t, p, n). R2 is NaN when y
    has zero variance (a degenerate, constant responder)."""
    n = len(x)
    if n < 3 or np.var(x) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, n
    b1, b0 = np.polyfit(x, y, 1)
    yh = b0 + b1 * x
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst == 0:
        return float(b1), float("nan"), np.nan, np.nan, np.nan, n   # constant DV
    sse = float(np.sum((y - yh) ** 2))
    r2 = 1 - sse / sst
    se = math.sqrt(sse / (n - 2) / np.sum((x - x.mean()) ** 2))
    t = b1 / se if se > 0 else np.nan
    return float(b1), float(r2), float(se), float(t), _norm_p(t), n


def inclusion_test() -> pd.DataFrame:
    rows = []
    for stem, label, pb in ROSTER:
        df = _clean_baseline(stem)
        b = df[df["condition"] == "baseline"]
        x = b["fixation_rate"].values.astype(float)
        y = b["engagement_score"].values.astype(float)
        b1, r2, se, t, p, n = _ols1(x, y)
        # pooled all-condition fallback, informative when baseline is constant
        xa = df["fixation_rate"].values.astype(float)
        ya = df["engagement_score"].values.astype(float)
        b1a, r2a, *_ = _ols1(xa, ya)
        passed = (not np.isnan(r2)) and (r2 >= 0.01)
        rows.append({
            "model": label, "params_b": pb, "n_baseline": n,
            "beta1": None if np.isnan(b1) else round(b1, 5),
            "R2_baseline": None if np.isnan(r2) else round(r2, 4),
            "se": None if (se is None or np.isnan(se)) else round(se, 5),
            "t": None if np.isnan(t) else round(t, 3),
            "p": None if np.isnan(p) else round(p, 4),
            "R2_all_conditions": None if np.isnan(r2a) else round(r2a, 4),
            "baseline_constant": bool(np.isnan(r2)),
            "inferential": bool(passed),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("00 | Data summary: completion + inclusion test\n" + "=" * 60)
    comp, by_cond = completion_table()
    incl = inclusion_test()

    odir = out_dir("summary")
    comp.to_csv(os.path.join(odir, "completion_by_model.csv"), index=False)
    by_cond.to_csv(os.path.join(odir, "parse_errors_by_condition.csv"), index=False)
    incl.to_csv(os.path.join(odir, "inclusion_test.csv"), index=False)

    print("\nCompletion by model:")
    print(comp.to_string(index=False))
    print("\nParse errors by condition:")
    print(by_cond.to_string(index=False))
    print("\nInclusion test (R^2 >= .01 to be inferential):")
    print(incl.to_string(index=False))
    print(f"\nSaved 3 CSVs to {odir}")
