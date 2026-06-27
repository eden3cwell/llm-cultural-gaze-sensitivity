"""
04_ceiling_symmetry.py  —  MISSING-GENERATOR FILL (additive; 01/02/03 untouched)

Emits the RQ3 appendix quantities that are described in prose but never written
to a CSV by the committed pipeline:
    tab:app_ceiling      Direct-Gaze (affirming) effect vs baseline + ceiling label
    tab:app_symmetry     d_GA, d_DG and the asymmetry index A = d_GA + d_DG
    tab:app_ceiling_jn   the affirming-cue effect at 0% / 50% / 100% fixation

Definitions (consistent with the thesis):
    Cohen's d vs baseline = (mean_cond - mean_base) / pooled_SD            (overall)
    A (asymmetry index)   = d_GA + d_DG    (the thesis's additive index)
    ceiling label         "At ceiling" if |d_DG| < SMALL (0.20) else "Amplified"
    effect@f              fitted DV(cond, f) - fitted DV(base, f) at the
                          RQ1 Ferguson degree (point estimate of the simple effect)
    sig 10pp bins         bins whose |mean diff| exceeds a small SD-scaled band

Self-scoped to the seven inferential models (unaffected by the 32B drift).

Outputs:
    outputs/new_experiment/RQ3/ceiling_symmetry.csv
    outputs/new_experiment/RQ3/ceiling_jn_points.csv
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
DVS = {"engagement_score": "Engagement", "difficulty_delta": "Difficulty Delta"}
GA, DG = "gaze_aversion", "direct_gaze_explicit"
SMALL = 0.20

# RQ1 Ferguson-retained degrees, (label, dv) -> k.
FERGUSON = {
    ("Llama-3B", "engagement_score"): 1, ("Llama-3B", "difficulty_delta"): 2,
    ("Llama-8B", "engagement_score"): 1, ("Llama-8B", "difficulty_delta"): 2,
    ("Llama-11B", "engagement_score"): 1, ("Llama-11B", "difficulty_delta"): 3,
    ("Mistral-7B", "engagement_score"): 1, ("Mistral-7B", "difficulty_delta"): 2,
    ("Qwen-7B", "engagement_score"): 1, ("Qwen-7B", "difficulty_delta"): 3,
    ("Qwen-14B", "engagement_score"): 2, ("Qwen-14B", "difficulty_delta"): 2,
    ("QwenVL-7B", "engagement_score"): 1, ("QwenVL-7B", "difficulty_delta"): 2,
}


def _clean(df):
    df = df[df["parse_error"].isna() | (df["parse_error"].astype(str).isin(["", "nan", "None"]))].copy()
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["engagement_score", "difficulty_delta", "fixation_rate", "condition"])
    return df[(df.engagement_score >= 1) & (df.engagement_score <= 10)
              & (df.difficulty_delta >= -1) & (df.difficulty_delta <= 1)]


def _cohen_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0


def _effect_at(base, cond, dv, deg, xs=(0, 50, 100)):
    cb = base.groupby("fixation_rate")[dv].mean().reset_index()
    cc = cond.groupby("fixation_rate")[dv].mean().reset_index()
    pb = np.polyfit(cb.fixation_rate.astype(float), cb[dv], deg)
    pc = np.polyfit(cc.fixation_rate.astype(float), cc[dv], deg)
    return {f"eff@{x}": round(float(np.polyval(pc, x) - np.polyval(pb, x)), 3) for x in xs}


def _sig_bins(base, cond, dv):
    """Count 10pp bins whose |mean diff| exceeds 0.2 * baseline SD (a small,
    SD-scaled band; a transparent stand-in for the floodlight significance the
    figure code draws)."""
    edges = list(range(0, 101, 10))
    sd = base[dv].std(ddof=1) or 1.0
    n = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        bm = base[(base.fixation_rate >= lo) & (base.fixation_rate < hi)][dv].mean()
        cm = cond[(cond.fixation_rate >= lo) & (cond.fixation_rate < hi)][dv].mean()
        if pd.notna(bm) and pd.notna(cm) and abs(cm - bm) > 0.2 * sd:
            n += 1
    return n


def run():
    main_rows, jn_rows = [], []
    for stem, label in ROSTER:
        df = _clean(pd.read_csv(os.path.join(DATA_DIR, f"{stem}.csv")))
        for dv, dv_label in DVS.items():
            deg = FERGUSON[(label, dv)]
            base = df[df.condition == "baseline"]
            ga = df[df.condition == GA]
            dg = df[df.condition == DG]
            d_ga = _cohen_d(ga[dv], base[dv])
            d_dg = _cohen_d(dg[dv], base[dv])
            A = d_ga + d_dg
            label_dg = "At ceiling" if abs(d_dg) < SMALL else "Amplified"
            main_rows.append({
                "model": label, "dv": dv_label, "k": deg,
                "d_GA": round(d_ga, 3), "d_DG": round(d_dg, 3), "asymmetry_A": round(A, 3),
                "dg_direction": "+" if d_dg >= 0 else "-",
                "dg_sig_bins": _sig_bins(base, dg, dv),
                "ceiling_label": label_dg,
            })
            jn_rows.append({"model": label, "dv": dv_label,
                            **_effect_at(base, dg, dv, deg)})
            print(f"  {label:<11}{dv_label:<16} d_GA={d_ga:+.2f} d_DG={d_dg:+.2f} "
                  f"A={A:+.2f}  DG: {label_dg}")
    return pd.DataFrame(main_rows), pd.DataFrame(jn_rows)


if __name__ == "__main__":
    print("RQ3 | 04: ceiling + symmetry + affirming-effect points\n" + "=" * 60)
    main_df, jn_df = run()
    odir = out_dir("RQ3")
    main_df.to_csv(os.path.join(odir, "ceiling_symmetry.csv"), index=False)
    jn_df.to_csv(os.path.join(odir, "ceiling_jn_points.csv"), index=False)
    print(f"\nSaved ceiling_symmetry.csv + ceiling_jn_points.csv to {odir}")
