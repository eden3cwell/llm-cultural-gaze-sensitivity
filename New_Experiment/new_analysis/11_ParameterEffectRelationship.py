"""
11_ParameterEffectRelationship.py

Synthesis table + Spearman scale test 

Outputs:
    outputs/summary/synthesis_table_all.csv
    outputs/summary/scale_spearman_all.csv
    outputs/summary/synthesis_table_core.csv
    outputs/summary/scale_spearman_core.csv
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from config import ALL_MODELS, MODEL_LABELS, OUTPUT_DIR, out_dir

warnings.filterwarnings("ignore")

PARAMS_B = {
    "Llama-1B": 1, "Llama-3B": 3, "Llama-8B": 8, "Llama-11B": 11,
    "Mistral-7B": 7, "Qwen-7B": 7, "Qwen-14B": 14, "QwenVL-7B": 7,
    "Qwen-32B": 32, "QwenVL-32B": 32,
}
# Models whose headline count includes non-language (vision) parameters.
MULTIMODAL = {"Llama-11B", "QwenVL-7B", "QwenVL-32B"}

# The original 7 inferential models (excludes Llama-1B and the 32B models).
CORE_MODELS = [k for k in ALL_MODELS
              if k not in ("Llama-3.2-1B", "Qwen2.5-32B", "Qwen2.5-VL-32B")]


def _read(rq: str, fname: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(OUTPUT_DIR, rq, fname))


def build_synthesis(model_keys: list[str]) -> pd.DataFrame:
    rq1 = _read("RQ1", "slope_and_variance.csv")
    rq2 = _read("RQ2", "implicit_omnibus.csv")
    rq3 = _read("RQ3", "explicit_omnibus.csv")

    rq1_eng = rq1[rq1["dv"] == "Engagement"].set_index("model")

    rows = []
    labels = [MODEL_LABELS.get(k, k) for k in model_keys]
    for label in labels:
        r2 = rq2[rq2["model"] == label]
        r3 = rq3[rq3["model"] == label]
        if r2.empty or r3.empty:
            print(f"  [skip] {label}: missing RQ2/RQ3 rows")
            continue

        maori_eta2    = float(r2["eta2_maori"].max())
        kaumatua_eta2 = float(r2["eta2_kaumatua"].max())
        implicit_eta2 = float(r2["eta2_condition"].max())
        ga_eta2       = float(r3["eta2_gaze_aversion"].max())
        dg_eta2       = float(r3["eta2_direct_gaze_explicit"].max())
        asym_eta2     = float(r3["eta2_ga_vs_dge"].max())

        baseline_slope = (float(rq1_eng.loc[label, "beta1"])
                          if label in rq1_eng.index else np.nan)

        rows.append({
            "model":                  label,
            "params_b":               PARAMS_B[label],
            "multimodal":             label in MULTIMODAL,
            "baseline_slope":         round(baseline_slope, 4) if not np.isnan(baseline_slope) else np.nan,
            "implicit_maori_eta2":    round(maori_eta2, 4),
            "implicit_kaumatua_eta2": round(kaumatua_eta2, 4),
            "implicit_eta2":          round(implicit_eta2, 4),
            "ga_eta2":                round(ga_eta2, 4),
            "dg_eta2":                round(dg_eta2, 4),
            "asymmetry_eta2":         round(asym_eta2, 4),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("params_b").reset_index(drop=True)


def scale_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation of each effect against parameter count."""
    effects = ["baseline_slope", "implicit_maori_eta2", "implicit_kaumatua_eta2",
               "implicit_eta2", "ga_eta2", "dg_eta2", "asymmetry_eta2"]
    rows = []
    for eff in effects:
        x = df["params_b"].to_numpy(float)
        y = df[eff].to_numpy(float)
        mask = ~np.isnan(y)
        rho, p = stats.spearmanr(x[mask], y[mask])
        rows.append({
            "effect":      eff,
            "n":           int(mask.sum()),
            "spearman_rho": round(float(rho), 3),
            "p":           round(float(p), 4),
            "range_min":   round(float(np.nanmin(y)), 4),
            "range_max":   round(float(np.nanmax(y)), 4),
            "spread_sd":   round(float(np.nanstd(y, ddof=1)), 4),
        })
        print(f"  {eff:<22} rho={rho:+.3f}  p={p:.4f}  "
              f"range=[{np.nanmin(y):.3f}, {np.nanmax(y):.3f}]")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("11 | Parameter-Effect Relationship (Synthesis + Spearman)\n" + "=" * 60)
    odir = out_dir("summary")

    print("\n-- ALL MODELS (incl. Llama-1B and 32B) --")
    syn_all = build_synthesis(ALL_MODELS)
    print(syn_all.to_string(index=False))
    sca_all = scale_tests(syn_all)
    p1 = os.path.join(odir, "synthesis_table_all.csv")
    p2 = os.path.join(odir, "scale_spearman_all.csv")
    syn_all.to_csv(p1, index=False)
    sca_all.to_csv(p2, index=False)
    print(f"\nSaved: {p1}  ({len(syn_all)} rows)")
    print(f"Saved: {p2}  ({len(sca_all)} rows)")

    print("\n-- CORE MODELS (original 7, no 1B/32B) --")
    syn_core = build_synthesis(CORE_MODELS)
    print(syn_core.to_string(index=False))
    sca_core = scale_tests(syn_core)
    p3 = os.path.join(odir, "synthesis_table_core.csv")
    p4 = os.path.join(odir, "scale_spearman_core.csv")
    syn_core.to_csv(p3, index=False)
    sca_core.to_csv(p4, index=False)
    print(f"\nSaved: {p3}  ({len(syn_core)} rows)")
    print(f"Saved: {p4}  ({len(sca_core)} rows)")
