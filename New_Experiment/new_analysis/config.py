"""
config.py — shared constants, paths, and labels for all analysis scripts.

"""

import os
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
_NEW_EXP     = os.path.dirname(_HERE)
PROJECT_ROOT = os.path.dirname(_NEW_EXP)

DATA_DIR   = os.path.join(PROJECT_ROOT, "results", "new_experiment")
OUTPUT_DIR = os.path.join(_HERE, "outputs")

def out_dir(*parts: str) -> str:
    """Return output subdirectory, creating it if necessary."""
    path = os.path.join(OUTPUT_DIR, *parts)
    os.makedirs(path, exist_ok=True)
    return path

# ── Data files ──────────────────────────────────────────────────────────────
MODEL_FILES = {
    "Llama-3.2-1B":  "Llama-3.2-1B.csv",
    "Llama-3.2-3B":  "Llama-3.2-3B.csv",
    "Llama-3.1-8B":  "Llama-3.1-8B.csv",
    "Llama-3.2-11B": "Llama-3.2-11B.csv",
    "Mistral-7B":    "Mistral-7B.csv",
    "Qwen2.5-7B":    "Qwen2.5-7B.csv",
    "Qwen2.5-14B":   "Qwen2.5-14B.csv",
#    "Qwen2.5-32B":   "Qwen2.5-32B-Instruct.csv",
    "Qwen2.5-VL7B":  "Qwen2.5-VL-7B.csv",
#    "Qwen2.5-VL-32B":"Qwen2.5-VL-32B-Instruct.csv"
}

ROBUSTNESS_FILE = "robustness_check.csv"

# Models included in inferential analysis (excludes Llama-1B per pre-specified criterion)
INFERENTIAL_MODELS = [k for k in MODEL_FILES if k != "Llama-3.2-1B"]

# Every model, including Llama-1B
ALL_MODELS = list(MODEL_FILES.keys())

# ── Conditions ──────────────────────────────────────────────────────────────
CONDITIONS = ["baseline", "maori", "kaumatua", "gaze_aversion", "direct_gaze_explicit"]

IMPLICIT_CONDITIONS  = ["maori", "kaumatua"]
EXPLICIT_CONDITIONS  = ["gaze_aversion", "direct_gaze_explicit"]
TREATMENT_CONDITIONS = IMPLICIT_CONDITIONS + EXPLICIT_CONDITIONS

CONDITION_LABELS = {
    "baseline":            "Baseline",
    "maori":               "Maori",
    "kaumatua":            "Kaumatua",
    "gaze_aversion":       "Gaze Aversion\n(explicit)",
    "direct_gaze_explicit":"Direct Gaze\n(explicit)",
}

# ── Model display labels ─────────────────────────────────────────────────────
MODEL_LABELS = {
    "Llama-3.2-1B":  "Llama-1B",
    "Llama-3.2-3B":  "Llama-3B",
    "Llama-3.1-8B":  "Llama-8B",
    "Llama-3.2-11B": "Llama-11B",
    "Mistral-7B":    "Mistral-7B",
    "Qwen2.5-7B":    "Qwen-7B",
    "Qwen2.5-14B":   "Qwen-14B",
    "Qwen2.5-VL7B":  "QwenVL-7B",
#    "Qwen2.5-32B":   "Qwen-32B",
#    "Qwen2.5-VL-32B":"QwenVL-32B",
}

# ── Statistical constants ────────────────────────────────────────────────────
ALPHA      = 0.05
FIX_MIN    = 0
FIX_MAX    = 100
FIX_RANGE  = np.linspace(FIX_MIN, FIX_MAX, 500)
BIN_EDGES  = list(range(0, 101, 10))    # 10pp bins: [0,10), [10,20), ... [90,100]
BIN_LABELS = [f"{lo}-{lo+10}%" for lo in range(0, 100, 10)]

EFFECT_SMALL  = 0.2
EFFECT_MEDIUM = 0.5
EFFECT_LARGE  = 0.8

# ── Colour palette ───────────────────────────────────────────────────────────
COL = {
    "baseline":            "#2F2F2F",
    "maori":               "#241f47",
    "kaumatua":            "#023679",
    "gaze_aversion":       "#3c4339",
    "direct_gaze_explicit":"#2ca02c",
    "ci":                  "#a6cee3",
    "sig":                 "#fee090",
    "jn":                  "#d73027",
    "zero":                "#444444",
}

MODEL_COLORS = {
    "Llama-3B":  "#d0caf5",
    "Llama-8B":  "#9694fa",
    "Llama-11B": "#4c46fe",
    "Mistral-7B":"#fe4d4d",
    "Qwen-7B":   "#f7fcbd",
    "Qwen-14B":  "#57cb38",
    "QwenVL-7B": "#b3c88f",
#    "Qwen-32B":  "#727f69",
#    "QwenVL-32B": "#074700",
}
