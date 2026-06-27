"""
RQ4 | Output 4: Cross-RQ synthesis table + scale (Spearman) test.

Assembles each inferential model's "cultural profile" by pulling effect sizes
from the RQ1-RQ3 result CSVs, attaches parameter count, and tests whether any
effect magnitude correlates monotonically with scale (Spearman). The scale test
is exploratory: n=7 with three models tied at 7B, so a null is non-evidence.

Columns assembled per model (engagement and difficulty kept separate where the
distinction matters):
    params_b            parameter count (B), from the model name
    baseline_slope      RQ1 degree-1 slope (engagement; the WEIRD-direction index)
    implicit_eta2       RQ2 omnibus condition eta^2 (max of the two DVs)
    ga_eta2             RQ3 gaze-aversion eta^2 (the override magnitude)
    dg_eta2             RQ3 direct-gaze eta^2 (the ceiling index)
    asymmetry_eta2      RQ3 GA-vs-DG contrast eta^2

Reads:
    RQ1/slope_inference_table.csv
    RQ2/omnibus_results.csv
    RQ3/omnibus_results.csv
Outputs:
    outputs/new_experiment/RQ4/synthesis_table.csv
    outputs/new_experiment/RQ4/scale_spearman.csv
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import INFERENTIAL_MODELS, MODEL_LABELS, OUTPUT_DIR, out_dir

warnings.filterwarnings("ignore")

# Parameter counts (B) by display label, taken from the model names.
PARAMS_B = {
    "Llama-3B": 3, "Llama-8B": 8, "Llama-11B": 11,
    "Mistral-7B": 7, "Qwen-7B": 7, "Qwen-14B": 14, "QwenVL-7B": 7,
}
# Models whose headline count includes non-language (vision) parameters.
MULTIMODAL = {"Llama-11B", "QwenVL-7B"}


def _read(rq, fname):
    return pd.read_csv(os.path.join(OUTPUT_DIR, rq, fname))


def build_synthesis() -> pd.DataFrame:
    rq1 = _read("RQ1", "slope_inference_table.csv")
    rq2 = _read("RQ2", "omnibus_results.csv")
    rq3 = _read("RQ3", "omnibus_results.csv")

    # RQ1 slope is per (model, dv); engagement degree-1 slope is the WEIRD index.
    rq1_eng = rq1[rq1["dv"] == "engagement_score"].set_index("model")

    rows = []
    labels = [MODEL_LABELS[k] for k in INFERENTIAL_MODELS]
    for label in labels:
        r2 = rq2[rq2["model"] == label]
        r3 = rq3[rq3["model"] == label]

        # take the larger of the two DVs for the omnibus-magnitude summaries
        implicit_eta2 = float(r2["eta2_condition"].max())
        ga_eta2       = float(r3["eta2_gaze_aversion"].max())
        dg_eta2       = float(r3["eta2_direct_gaze_explicit"].max())
        asym_eta2     = float(r3["eta2_ga_vs_dge"].max())

        baseline_slope = (float(rq1_eng.loc[label, "beta1"])
                          if label in rq1_eng.index else np.nan)

        rows.append({
            "model":          label,
            "params_b":       PARAMS_B[label],
            "multimodal":     label in MULTIMODAL,
            "baseline_slope": round(baseline_slope, 4),
            "implicit_eta2":  round(implicit_eta2, 4),
            "ga_eta2":        round(ga_eta2, 4),
            "dg_eta2":        round(dg_eta2, 4),
            "asymmetry_eta2": round(asym_eta2, 4),
        })
    df = pd.DataFrame(rows)
    # order by parameter count for readability
    return df.sort_values("params_b").reset_index(drop=True)


def scale_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation of each effect against parameter count.
    Exploratory: n=7 with ties at 7B; report rho and p but treat a null as
    non-evidence of no relationship."""
    effects = ["baseline_slope", "implicit_eta2", "ga_eta2",
               "dg_eta2", "asymmetry_eta2"]
    rows = []
    for eff in effects:
        x = df["params_b"].to_numpy(float)
        y = df[eff].to_numpy(float)
        mask = ~np.isnan(y)
        rho, p = stats.spearmanr(x[mask], y[mask])
        rows.append({
            "effect":   eff,
            "n":        int(mask.sum()),
            "spearman_rho": round(float(rho), 3),
            "p":        round(float(p), 4),
            "range_min": round(float(np.nanmin(y)), 4),
            "range_max": round(float(np.nanmax(y)), 4),
            "spread_sd": round(float(np.nanstd(y, ddof=1)), 4),
        })
        print(f"  {eff:<16} rho={rho:+.3f}  p={p:.4f}  "
              f"range=[{np.nanmin(y):.3f}, {np.nanmax(y):.3f}]")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("RQ4 | Output 4: Synthesis + Scale Test\n" + "=" * 60)
    syn = build_synthesis()
    print(syn.to_string(index=False))
    print()
    sca = scale_tests(syn)

    p1 = os.path.join(out_dir("RQ4"), "synthesis_table.csv")
    p2 = os.path.join(out_dir("RQ4"), "scale_spearman.csv")
    syn.to_csv(p1, index=False)
    sca.to_csv(p2, index=False)
    print(f"\nSaved: {p1}  ({len(syn)} rows)")
    print(f"Saved: {p2}  ({len(sca)} rows)")