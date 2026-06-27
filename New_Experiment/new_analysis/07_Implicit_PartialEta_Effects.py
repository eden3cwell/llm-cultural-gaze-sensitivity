"""
07_Implicit_PartialEta_Effects.py

Output: outputs/RQ2/implicit_omnibus.csv
"""

from __future__ import annotations

import os
import sys
import warnings
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(__file__))
from config import (DATA_DIR, MODEL_FILES, MODEL_LABELS, ALL_MODELS,
                    IMPLICIT_CONDITIONS, ALPHA, out_dir)

warnings.filterwarnings("ignore")

DVS = ["engagement_score", "difficulty_delta"]

FORMULAS = {
    1: "{dv} ~ fixation_rate",
    2: "{dv} ~ fixation_rate + I(fixation_rate**2)",
    3: "{dv} ~ fixation_rate + I(fixation_rate**2) + I(fixation_rate**3)",
}

FERGUSON_THRESHOLD = 0.04   # minimum practical R^2 gain (Ferguson, 2009)


def _load_clean(model_key: str) -> pd.DataFrame:
    """All conditions, cleaned (filtering to specific conditions is the
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


def _poly_terms(k: int) -> str:
    """Formula terms 'fixation_rate + I(fixation_rate ** 2) + ...' up to degree k."""
    terms = ["fixation_rate"] + [f"I(fixation_rate ** {j})" for j in range(2, k + 1)]
    return " + ".join(terms)


def _compare(rss_r: float, rss_f: float, p_r: int, p_f: int,
             n: int, tss: float) -> dict:
    """Nested F-test + effect sizes comparing reduced to full OLS model.
    No statsmodels/scipy function outputs partial eta^2 or dR2 directly, so
    these are derived by hand from the two models' residual sums of squares."""
    df1 = p_f - p_r
    df2 = n - p_f
    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    p = float(scipy_stats.f.sf(F, df1, df2))
    eta2 = (rss_r - rss_f) / rss_r                 # partial eta-squared
    dR2 = (rss_r - rss_f) / tss                    # incremental R^2 over reduced
    return {"F": round(F, 3), "df1": df1, "df2": df2, "p": round(p, 5),
            "eta2": round(eta2, 4), "dR2": round(dR2, 4)}


def _design_fit(df: pd.DataFrame, dv: str, k: int, conds: list[str]):
    """Fit reduced (poly only), mid (poly + condition), and full (poly *
    condition) OLS models via the statsmodels formula API. The first entry
    in `conds` is the reference category, matching the original convention."""
    d = df[df["condition"].isin(conds)]
    poly = _poly_terms(k)
    ref = conds[0]
    cond_term = f"C(condition, Treatment('{ref}'))"

    m_reduced = smf.ols(f"{dv} ~ {poly}", data=d).fit()
    m_mid     = smf.ols(f"{dv} ~ {cond_term} + {poly}", data=d).fit()
    m_full    = smf.ols(f"{dv} ~ {cond_term} * ({poly})", data=d).fit()

    n   = int(m_full.nobs)
    tss = float(m_full.centered_tss)
    return m_reduced, m_mid, m_full, n, tss


def run_omnibus(df: pd.DataFrame, dv: str, k: int) -> dict:
    """Omnibus, interaction, per-label, and within-culture tests for one model/DV."""
    conds = ["baseline"] + list(IMPLICIT_CONDITIONS)         # baseline, maori, kaumatua
    Mp, Mm, Mf, n, tss = _design_fit(df, dv, k, conds)

    omni  = _compare(Mp.ssr, Mf.ssr, len(Mp.params), len(Mf.params), n, tss)  # any condition effect
    inter = _compare(Mm.ssr, Mf.ssr, len(Mm.params), len(Mf.params), n, tss)  # moderation only

    out = {
        "degree": k, "n": n, "r2_full": round(1 - Mf.ssr / tss, 4),
        "F_condition": omni["F"], "df_condition": f"{omni['df1']},{omni['df2']}",
        "p_condition": omni["p"], "eta2_condition": omni["eta2"],
        "dR2_condition": omni["dR2"], "sig_condition": bool(omni["p"] < ALPHA),
        "F_interaction": inter["F"], "df_interaction": f"{inter['df1']},{inter['df2']}",
        "p_interaction": inter["p"], "eta2_interaction": inter["eta2"],
    }

    # Per-label: baseline vs each implicit condition separately
    for cond in IMPLICIT_CONDITIONS:
        Mp2, _Mm2, Mf2, n2, tss2 = _design_fit(df, dv, k, ["baseline", cond])
        res = _compare(Mp2.ssr, Mf2.ssr, len(Mp2.params), len(Mf2.params), n2, tss2)
        out[f"F_{cond}"]    = res["F"]
        out[f"p_{cond}"]    = res["p"]
        out[f"eta2_{cond}"] = res["eta2"]
        out[f"dR2_{cond}"]  = res["dR2"]

    # Within-culture contrast: kaumatua vs maori (no baseline)
    Mp3, _Mm3, Mf3, n3, tss3 = _design_fit(df, dv, k, list(IMPLICIT_CONDITIONS))
    kvm = _compare(Mp3.ssr, Mf3.ssr, len(Mp3.params), len(Mf3.params), n3, tss3)
    out["F_kaum_vs_maori"]    = kvm["F"]
    out["df_kaum_vs_maori"]   = f"{kvm['df1']},{kvm['df2']}"
    out["p_kaum_vs_maori"]    = kvm["p"]
    out["eta2_kaum_vs_maori"] = kvm["eta2"]
    return out


def run() -> pd.DataFrame:
    rows = []
    for model_key in ALL_MODELS:
        path = os.path.join(DATA_DIR, MODEL_FILES[model_key])
        if not os.path.exists(path):
            print(f"  [MISSING] {MODEL_FILES[model_key]}")
            continue
        data  = _load_clean(model_key)
        label = MODEL_LABELS.get(model_key, model_key)
        baseline = data[data["condition"] == "baseline"]

        for dv in DVS:
            k = ferguson_degree_for(dv, baseline)
            result = run_omnibus(data, dv, k)
            rows.append({"model": label, "dv": dv, **result})
            print(f"  {label:<12} {dv:<18} k={result['degree']}  "
                  f"eta2_cond={result['eta2_condition']:<7} "
                  f"M={result['eta2_maori']:<7} K={result['eta2_kaumatua']:<7} "
                  f"int={result['eta2_interaction']:<7} "
                  f"KvM eta2={result['eta2_kaum_vs_maori']}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("07 | RQ2: Implicit-Framing Omnibus (Partial Eta^2)\n" + "=" * 60)
    results = run()
    path = os.path.join(out_dir("RQ2"), "implicit_omnibus.csv")
    results.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(results)} rows)")
