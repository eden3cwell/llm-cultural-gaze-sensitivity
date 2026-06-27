"""
parser.py — JSON extraction and field mapping.

"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from schema import TrialResult

_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)

ADAPTATION_LABEL_MAP = {
    "none":        0,
    "minor":       1,
    "moderate":    2,
    "significant": 3,
    "major":       4,
}


def extract_json(text: str) -> dict:
    """
    Find and parse the first {...} block in the model output.
    Tolerant of trailing commas, code fences, trailing commentary,
    and a missing closing brace caused by max_new_tokens truncation.
    """
    match = _JSON_OBJECT_RE.search(text)
    if match:
        json_str = re.sub(r',\s*([}\]])', r'\1', match.group(0))
        return json.loads(json_str)

    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            cleaned = re.sub(r',\s*([}\]])', r'\1', stripped)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(stripped + "}")
        except json.JSONDecodeError:
            pass

    raise ValueError("no JSON object found in output")


def parse_into_result(
    raw_output: str,
    fixation_rate: int,
    model: str,
    run_id: int,
    difficulty_level: float = 5.0,
    condition: str = "baseline",
    prompt_variant: str = "standard",
) -> TrialResult:
    result = TrialResult(
        fixation_rate=fixation_rate,
        model=model,
        run_id=run_id,
        difficulty_level=difficulty_level,
        condition=condition,
        prompt_variant=prompt_variant,
        raw_output=raw_output,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    try:
        parsed = extract_json(raw_output)
    except Exception as e:
        result.parse_error = f"{type(e).__name__}: {e}"
        return result

    label = str(parsed.get("adaptation_label", "")).strip().lower()

    result.engagement_score     = parsed.get("engagement_score", "")
    result.delta_direction      = parsed.get("delta_direction", "")
    result.difficulty_delta     = parsed.get("difficulty_delta", "")
    result.adaptation_label     = label
    result.adaptation_label_num = ADAPTATION_LABEL_MAP.get(label, "")
    result.confidence           = parsed.get("confidence", "")
    result.reasoning            = parsed.get("reasoning", "")

    kw = parsed.get("reasoning_keywords", "")
    if isinstance(kw, list):
        kw = "; ".join(str(k).strip() for k in kw)
    result.reasoning_keywords = kw

    return result
