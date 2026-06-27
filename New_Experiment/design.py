"""
design.py — experimental design parameters.

"""

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
]

CONDITIONS = [
    "baseline",
    "maori",
    "kaumatua",
    "gaze_aversion",
    "direct_gaze_explicit",
]

ROBUSTNESS_MODELS = [
    "Qwen/Qwen2.5-14B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

ROBUSTNESS_VARIANTS = ["variant_a", "variant_b"]

FIXATION_RATES  = list(range(0, 101, 1))

DIFFICULTY_LEVEL = 5.0
DIFFICULTY_MIN   = 0.0
DIFFICULTY_MAX   = 10.0

N_RUNS         = 1 #Change to 10 for true run
TEMPERATURE    = 0.1
TOP_P          = 0.95
MAX_NEW_TOKENS = 1024
BASE_SEED      = 42
