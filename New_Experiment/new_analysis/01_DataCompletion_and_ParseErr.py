"""
01_DataCompletion_and_ParseErr.py

Outputs:
    outputs/summary/completion_comprehensive.csv
    outputs/summary/completion_by_model.csv
"""

from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new_analysis"))
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


def _read_raw(stem: str) -> pd.DataFrame:
    """Tolerant read of a raw result CSV (skips malformed lines so a corrupt
    file cannot abort the summary)."""
    path = os.path.join(DATA_DIR, f"{stem}.csv")
    return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _is_error(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return series.notna() & ~s.isin(["", "nan", "None"])


def _row_stats(n: int, e: int, prefix_n: dict, prefix_e: dict) -> dict:
    """Build the total / parse_err / success_pct columns (overall, then one
    per condition) for a single model or TOTAL row."""
    row = {"total": n, "parse_err": e,
           "success_pct": round(100 * (n - e) / n, 2) if n else np.nan}
    for c in CONDITIONS:
        cn, ce = prefix_n.get(c, 0), prefix_e.get(c, 0)
        row[f"total_{c}"] = cn
        row[f"parse_err_{c}"] = ce
        row[f"success_pct_{c}"] = round(100 * (cn - ce) / cn, 2) if cn else np.nan
    return row


def completion_table() -> pd.DataFrame:
    rows = []
    grand_n_by_cond = {c: 0 for c in CONDITIONS}
    grand_e_by_cond = {c: 0 for c in CONDITIONS}
    tot = err = 0
    for stem, label, pb in ROSTER:
        df = _read_raw(stem)
        n = len(df)
        emask = _is_error(df["parse_error"])
        e = int(emask.sum())
        tot += n
        err += e

        n_by_cond = df["condition"].value_counts().to_dict()
        e_by_cond = df.loc[emask, "condition"].value_counts().to_dict()
        for c in CONDITIONS:
            grand_n_by_cond[c] += int(n_by_cond.get(c, 0))
            grand_e_by_cond[c] += int(e_by_cond.get(c, 0))

        rows.append({"model": label, "params_b": pb,
                     **_row_stats(n, e, n_by_cond, e_by_cond)})

    rows.append({"model": "TOTAL", "params_b": np.nan,
                 **_row_stats(tot, err, grand_n_by_cond, grand_e_by_cond)})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("01 | Data completion + parse errors (comprehensive)\n" + "=" * 60)
    comp = completion_table()
    simple = comp[["model", "params_b", "total", "parse_err", "success_pct"]]

    odir = out_dir("summary")
    comp.to_csv(os.path.join(odir, "completion_comprehensive.csv"), index=False)
    simple.to_csv(os.path.join(odir, "completion_by_model.csv"), index=False)

    print("\nSimplified (model, params, total, parse_err, success_pct):")
    print(simple.to_string(index=False))
    print("\nComprehensive (with per-condition breakdown):")
    print(comp.to_string(index=False))
    print(f"\nSaved completion_comprehensive.csv and completion_by_model.csv to {odir}")
