"""
RQ2 | Output 3: Binned raw mean differences + overall aggregate, both DVs.

Primary magnitude metric (per RQ2 decisions): RAW MEAN DIFFERENCE per 10pp
fixation bin, in DV units (engagement points; delta units). This is immune to
the small-within-bin-variance inflation that makes binned Cohen's d exceed
|1| meaninglessly, and it carries direction.

Also retains an OVERALL aggregate per model x condition, reported as a labelled
summary only (it averages over fixation and so understates sign-flipping
effects). Overall is given as both mean difference and Cohen's d for reference.

Outputs:
    outputs/new_experiment/RQ2/binned_mean_diff.csv
    outputs/new_experiment/RQ2/overall_aggregate.csv
"""

from __future__ import annotations

import sys
import os
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import (INFERENTIAL_MODELS, IMPLICIT_CONDITIONS,
                    MODEL_LABELS, BIN_EDGES, BIN_LABELS, out_dir)
from loaders import load_all

warnings.filterwarnings("ignore")

DVS = ["engagement_score", "difficulty_delta"]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    psd = np.sqrt(((n_a - 1) * va + (n_b - 1) * vb) / (n_a + n_b - 2))
    return 0.0 if psd == 0 else float((a.mean() - b.mean()) / psd)


def run_binned(dfs: dict) -> pd.DataFrame:
    rows = []
    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        dm = dfs[model_key].copy()
        label = MODEL_LABELS[model_key]
        dm["bin"] = pd.cut(dm["fixation_rate"], bins=BIN_EDGES,
                           labels=BIN_LABELS, right=False)
        baseline = dm[dm["condition"] == "baseline"]

        for cond in IMPLICIT_CONDITIONS:
            treat = dm[dm["condition"] == cond]
            for b in BIN_LABELS:
                bl = baseline[baseline["bin"] == b]
                tr = treat[treat["bin"] == b]
                row = {"model": label, "condition": cond, "bin": b,
                       "n_baseline": len(bl), "n_treatment": len(tr)}
                for dv in DVS:
                    if len(bl) and len(tr):
                        row[f"meandiff_{dv}"] = round(
                            float(tr[dv].mean() - bl[dv].mean()), 4)
                    else:
                        row[f"meandiff_{dv}"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def run_overall(dfs: dict) -> pd.DataFrame:
    rows = []
    for model_key in INFERENTIAL_MODELS:
        if model_key not in dfs:
            continue
        dm = dfs[model_key]
        label = MODEL_LABELS[model_key]
        baseline = dm[dm["condition"] == "baseline"]
        for cond in IMPLICIT_CONDITIONS:
            treat = dm[dm["condition"] == cond]
            if len(treat) == 0:
                continue
            row = {"model": label, "condition": cond}
            for dv in DVS:
                row[f"meandiff_{dv}"] = round(
                    float(treat[dv].mean() - baseline[dv].mean()), 4)
                row[f"d_{dv}"] = round(
                    cohens_d(treat[dv].values, baseline[dv].values), 4)
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("RQ2 | Output 3: Binned mean differences + overall\n" + "=" * 60)
    dfs = load_all()

    binned = run_binned(dfs)
    pb = os.path.join(out_dir("RQ2"), "binned_mean_diff.csv")
    binned.to_csv(pb, index=False)
    print(f"Saved: {pb}  ({len(binned)} rows)")

    overall = run_overall(dfs)
    po = os.path.join(out_dir("RQ2"), "overall_aggregate.csv")
    overall.to_csv(po, index=False)
    print(f"Saved: {po}  ({len(overall)} rows)")