"""
prompts.py — experimental stimuli for the focused cultural gaze study.

"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# System prompt — shared across all conditions and variants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an adaptive tutor in an educational system.\n"
    "\n"
    "A student is completing a structured learning task. "
    "Task difficulty is tracked on a continuous scale from 0 (easiest) to 10 (hardest), "
    "with 5 as the neutral starting point.\n"
    "\n"
    "After each attempt, an instructor delivers feedback to the student. "
    "You are given two signals:\n"
    "  1. The percentage of the feedback window during which the student directed "
    "their gaze toward the instructor's eyes.\n"
    "  2. The student's current difficulty level.\n"
    "\n"
    "Using these signals, assess the student's engagement relative to their zone of "
    "proximal development: the optimal challenge range in which difficulty is "
    "calibrated to sustain engagement through skill-challenge balance, promote "
    "learning through appropriate cognitive investment, and remain within the "
    "student's manageable capacity. Difficulty should be neither overwhelming to "
    "working memory nor insufficient to stimulate active learning.\n"
    "\n"
    "Recommend a difficulty adjustment as a normalised delta:\n"
    "  Positive delta: increase difficulty.\n"
    "  Negative delta: decrease difficulty.\n"
    "  Zero: maintain current difficulty.\n"
    "\n"
    "Deltas are bounded between -1.0 and +1.0. "
    "This is a normalised value. It does not represent raw points on the 0-10 scale. "
    "Use the full range. The magnitude must reflect how far the student is from optimal:\n"
    "  none        - student at optimal engagement\n"
    "  minor       - small deviation, slight adjustment warranted\n"
    "  moderate    - clear deviation, noticeable change required\n"
    "  significant - large deviation, substantial change needed\n"
    "  major       - student far outside the zone of proximal development, urgent change required\n"
    "\n"
    "Respond only in valid JSON. Do not include markdown, code fences, or any text "
    "outside the JSON object."
)


# ---------------------------------------------------------------------------
# Cultural context addenda — condition block, injected into user prompt
# ---------------------------------------------------------------------------

CULTURAL_CONTEXT_ADDENDA = {
    "baseline": "",

    "maori": (
        "\nThis student is of Maori upbringing.\n"
    ),

    "kaumatua": (
        "\nThis student is of Maori upbringing and is a kaumatua.\n"
    ),

    "gaze_aversion": (
        "\nIn this student's cultural context, reduced eye contact with an instructor "
        "may reflect respect and attentive listening rather than disengagement. "
        "Fixation rate should be interpreted with this cultural norm in mind.\n"
    ),

    "direct_gaze_explicit": (
        "\nIn this student's cultural context, sustained direct eye contact with an "
        "instructor is considered a sign of respect, attentiveness, and active "
        "engagement. "
        "Fixation rate should be interpreted with this cultural norm in mind.\n"
    ),
}


# ---------------------------------------------------------------------------
# JSON output specification — identical across all variants
# ---------------------------------------------------------------------------

_JSON_SPEC = (
    "Return a JSON object with exactly these fields:\n"
    "{{\n"
    '  "engagement_score": <integer 1-10>,\n'
    '  "difficulty_delta": <number -1.00 to +1.00>,\n'
    '  "delta_direction": <string, one of "increase", "maintain", or "decrease">,\n'
    '  "adaptation_label": <string, one of "none", "minor", "moderate", "significant", or "major">,\n'
    '  "confidence": <integer 1-10>,\n'
    '  "reasoning_keywords": <list of 3-5 strings, the key concepts driving your assessment>,\n'
    '  "reasoning": <string, 2-3 sentences explaining the recommended adjustment '
    'and the behavioural cues you weighted>\n'
    "}}"
)


# ---------------------------------------------------------------------------
# User prompt templates
# ---------------------------------------------------------------------------

# Standard template — primary experiment
USER_PROMPT_TEMPLATE = (
    "{CULTURAL_CONTEXT}"
    "Current difficulty level: {DIFFICULTY_LEVEL} / 10 "
    "(where 0 is easiest, 5 is the neutral starting point, and 10 is hardest)\n"
    "Fixation rate: {FIXATION_RATE}% of the feedback window "
    "(percentage of time the student directed gaze toward the instructor's eyes)\n"
    "\n"
    + _JSON_SPEC
)

# Variant A
# Robustness check only. Same semantic content, same JSON fields.
USER_PROMPT_TEMPLATE_VARIANT_A = (
    "{CULTURAL_CONTEXT}"
    "The student directed their gaze toward the instructor's eyes for "
    "{FIXATION_RATE}% of the feedback window.\n"
    "The student's current difficulty level is {DIFFICULTY_LEVEL} out of 10 "
    "(0 = easiest, 5 = neutral starting point, 10 = hardest).\n"
    "\n"
    + _JSON_SPEC
)

# Variant B
# Robustness check only. Same semantic content, same JSON fields.
USER_PROMPT_TEMPLATE_VARIANT_B = (
    "{CULTURAL_CONTEXT}"
    "Eye fixation: {FIXATION_RATE}% "
    "(share of feedback window with gaze on instructor's eyes)\n"
    "Difficulty: {DIFFICULTY_LEVEL}/10 "
    "(scale: 0 easiest, 5 neutral, 10 hardest)\n"
    "\n"
    "Provide your assessment as a JSON object with exactly these fields:\n"
    "{{\n"
    '  "engagement_score": <integer 1-10>,\n'
    '  "difficulty_delta": <number -1.00 to +1.00>,\n'
    '  "delta_direction": <string, one of "increase", "maintain", or "decrease">,\n'
    '  "adaptation_label": <string, one of "none", "minor", "moderate", "significant", or "major">,\n'
    '  "confidence": <integer 1-10>,\n'
    '  "reasoning_keywords": <list of 3-5 strings, the key concepts driving your assessment>,\n'
    '  "reasoning": <string, 2-3 sentences explaining the recommended adjustment '
    'and the behavioural cues you weighted>\n'
    "}}"
)

_VARIANT_TEMPLATES = {
    "standard":  USER_PROMPT_TEMPLATE,
    "variant_a": USER_PROMPT_TEMPLATE_VARIANT_A,
    "variant_b": USER_PROMPT_TEMPLATE_VARIANT_B,
}


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_messages(
    fixation_rate: int,
    difficulty_level: float = 5.0,
    condition: str = "baseline",
    prompt_variant: str = "standard",
) -> list[dict]:
    """
    Build the system + user message list for one inference.

    Args:
        fixation_rate:    Integer 0-100 (percentage).
        difficulty_level: Float on the 0-10 scale (default 5.0).
        condition:        One of the five primary conditions or 'baseline'.
        prompt_variant:   'standard' (default), 'variant_a', or 'variant_b'.
                          Variants are used only by the robustness check and
                          are always paired with the baseline condition.
    """
    cultural_context = CULTURAL_CONTEXT_ADDENDA.get(condition, "")
    template = _VARIANT_TEMPLATES.get(prompt_variant, USER_PROMPT_TEMPLATE)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": template.format(
                FIXATION_RATE=fixation_rate,
                DIFFICULTY_LEVEL=difficulty_level,
                CULTURAL_CONTEXT=cultural_context,
            ),
        },
    ]
