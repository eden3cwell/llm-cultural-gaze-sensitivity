"""
trace_pull.py  —  MISSING-GENERATOR FILL (additive; new `qualitative/` folder)

Makes the Discussion's reasoning-trace examples reproducible. The committed
pipeline never touches the `reasoning` / `reasoning_keywords` columns, so the
qualitative cases quoted in the thesis cannot currently be regenerated or
audited. This script pulls, for a list of (model, condition, fixation) cases,
the model's reasoning text alongside its engagement / difficulty outputs AND the
same model's MATCHED BASELINE outputs at the identical fixation rate — exactly
the matched comparison the Discussion relies on.

Edit CASES below to the cells you quote. Running with no edits dumps the
illustrative cases referenced in the thesis as a starting point.

Self-scoped per requested model (unaffected by the 32B drift).

Outputs:
    outputs/new_experiment/qualitative/trace_examples.csv   (+ console print)
"""

from __future__ import annotations

import os
import sys
import textwrap
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
from config import DATA_DIR, out_dir

LABEL_TO_STEM = {
    "Llama-1B": "Llama-3.2-1B", "Llama-3B": "Llama-3.2-3B", "Llama-8B": "Llama-3.1-8B",
    "Llama-11B": "Llama-3.2-11B", "Mistral-7B": "Mistral-7B", "Qwen-7B": "Qwen2.5-7B",
    "Qwen-14B": "Qwen2.5-14B", "QwenVL-7B": "Qwen2.5-VL-7B",
}

# (model label, condition, fixation%) cases quoted in the Discussion.
CASES = [
    ("Llama-8B", "kaumatua", 1),
    ("Qwen-14B", "kaumatua", 2),
    ("Mistral-7B", "kaumatua", 9),
    ("QwenVL-7B", "kaumatua", 2),
    ("Mistral-7B", "gaze_aversion", 1),
    ("Llama-11B", "gaze_aversion", 1),
    ("Qwen-14B", "gaze_aversion", 2),
]


def _clean(df):
    df = df[df["parse_error"].isna() | (df["parse_error"].astype(str).isin(["", "nan", "None"]))].copy()
    for c in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["engagement_score", "difficulty_delta", "fixation_rate", "condition"])


def _cell(df, cond, fix):
    s = df[(df.condition == cond) & (df.fixation_rate == fix)]
    return s


def run():
    rows = []
    for label, cond, fix in CASES:
        stem = LABEL_TO_STEM[label]
        df = _clean(pd.read_csv(os.path.join(DATA_DIR, f"{stem}.csv")))
        cond_cell = _cell(df, cond, fix)
        base_cell = _cell(df, "baseline", fix)
        if len(cond_cell) == 0:
            print(f"  [skip] {label} {cond} @{fix}% — no rows")
            continue
        # representative reasoning: the most common keyword string's first row
        reasoning = str(cond_cell.iloc[0].get("reasoning", "")).strip()
        row = {
            "model": label, "condition": cond, "fixation": fix,
            "eng_cond": round(cond_cell.engagement_score.mean(), 2),
            "eng_base": round(base_cell.engagement_score.mean(), 2) if len(base_cell) else None,
            "delta_cond": round(cond_cell.difficulty_delta.mean(), 3),
            "delta_base": round(base_cell.difficulty_delta.mean(), 3) if len(base_cell) else None,
            "reasoning": reasoning,
        }
        rows.append(row)
        print(f"\n● {label}  {cond} @ {fix}% fixation  (n={len(cond_cell)} runs)")
        print(f"   engagement: cond {row['eng_cond']}  vs baseline {row['eng_base']}")
        print(f"   difficulty: cond {row['delta_cond']}  vs baseline {row['delta_base']}")
        print("   reasoning : " + textwrap.fill(reasoning, 88, subsequent_indent=" " * 15))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Qualitative | reasoning-trace pull (matched to baseline)\n" + "=" * 60)
    res = run()
    path = os.path.join(out_dir("qualitative"), "trace_examples.csv")
    res.to_csv(path, index=False)
    print(f"\nSaved: {path}  ({len(res)} rows)")
