"""
schema.py — output schema and TrialResult dataclass.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CSV_FIELDS = [
    "condition",
    "fixation_rate",
    "difficulty_level",
    "model",
    "run_id",
    "prompt_variant",
    "engagement_score",
    "difficulty_delta",    
    "delta_direction",
    "adaptation_label",
    "adaptation_label_num",
    "reasoning_keywords",
    "confidence",
    "reasoning",
    "timestamp",
    "parse_error",
]


@dataclass
class TrialResult:
    fixation_rate: int
    model: str
    run_id: int
    condition: str = "baseline"
    difficulty_level: float = 5.0
    prompt_variant: str = "standard"
    engagement_score: Any = ""
    difficulty_delta: Any = ""
    delta_direction: Any = ""
    adaptation_label: Any = ""
    adaptation_label_num: Any = ""
    reasoning_keywords: Any = ""
    confidence: Any = ""
    reasoning: Any = ""
    timestamp: str = ""
    parse_error: str = ""
    raw_output: str = field(default="", repr=False)

    def to_row(self) -> dict:
        return {k: getattr(self, k) for k in CSV_FIELDS}
