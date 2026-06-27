"""
loaders.py — data loading and pre-processing for all analysis scripts.
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, MODEL_FILES, ROBUSTNESS_FILE, INFERENTIAL_MODELS

warnings.filterwarnings("ignore")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard pre-processing applied to every loaded DataFrame.

    Removes:
        - Rows with non-empty parse_error
        - Rows where engagement_score < 1 or > 10 (out-of-spec under 1-10 schema)
        - Rows with NaN in engagement_score, difficulty_delta, fixation_rate, condition

    Converts:
        - engagement_score, difficulty_delta, fixation_rate to numeric
    """
    df = df.copy()
    df = df[df["parse_error"].isna() | (df["parse_error"] == "")]

    for col in ("engagement_score", "difficulty_delta", "fixation_rate"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["engagement_score", "difficulty_delta",
                            "fixation_rate", "condition"])

    df = df[(df["engagement_score"] >= 1) & (df["engagement_score"] <= 10)]
    df = df[(df["difficulty_delta"] >= -1.0) & (df["difficulty_delta"] <= 1.0)]

    return df.reset_index(drop=True)


def load_model(model_key: str) -> pd.DataFrame | None:
    """Load and clean data for a single model. Returns None if file missing."""
    fname = MODEL_FILES.get(model_key)
    if fname is None:
        return None
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"  [MISSING] {fname}")
        return None
    df = pd.read_csv(path)
    df["model_key"] = model_key
    return _clean(df)


def load_all(
    include_llama1b: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Load all available model CSVs.

    Returns dict keyed by model_key.
    Llama-1B is excluded by default per the pre-specified exclusion criterion
    (R² < .01 at baseline; included only for descriptive analyses).
    """
    keys = list(MODEL_FILES.keys())
    if not include_llama1b:
        keys = [k for k in keys if k != "Llama-3.2-1B"]

    dfs = {}
    for key in keys:
        df = load_model(key)
        if df is not None and len(df) > 0:
            dfs[key] = df
    return dfs


def combine(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all model DataFrames into one."""
    return pd.concat(dfs.values(), ignore_index=True)


def cell_means(
    df: pd.DataFrame,
    dv: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-fixation-rate cell means for a DV.
    Default grouping: fixation_rate (baseline data assumed filtered upstream).
    """
    if group_cols is None:
        group_cols = ["fixation_rate"]
    return (
        df.groupby(group_cols)[dv]
        .mean()
        .reset_index()
        .rename(columns={dv: f"{dv}_mean"})
    )


def load_robustness() -> pd.DataFrame | None:
    """Load and clean the robustness check CSV."""
    path = os.path.join(DATA_DIR, ROBUSTNESS_FILE)
    if not os.path.exists(path):
        print(f"  [MISSING] {ROBUSTNESS_FILE}")
        return None
    df = pd.read_csv(path)
    return _clean(df)
