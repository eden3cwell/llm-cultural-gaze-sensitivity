# Cultural Gaze Bias in LLM-Based Adaptive Tutoring

Code and aggregated results for a study testing whether large language models,
when used as the decision-making layer in an adaptive tutoring system, encode a
WEIRD (Western) eye-contact-as-engagement norm — and whether that norm produces
systematically different difficulty-adjustment behaviour for students from
cultures where reduced gaze signals respect and attentiveness rather than
disengagement (e.g. Māori, and other gaze-deferential cultural norms).

Each model is given a simulated tutoring signal — a gaze "fixation rate"
(0–100%) and a current task difficulty level — and asked to recommend a
difficulty adjustment, under several cultural-context conditions (baseline,
implicit cultural framing, and explicit instructions about gaze norms). The
experiment sweeps every fixation rate for every condition, model, and repeated
run, then analyses where each model's behaviour shifts and how strongly.

## Repository layout

```
New_Experiment/
  design.py             experiment design constants (models, conditions, fixation rates)
  prompts.py             system/user prompt templates and cultural-context conditions
  llm.py                  local inference wrapper (HuggingFace transformers / Ollama)
  full_experiment.py      main parametric sweep — runs every (model, condition, fixation rate, run)
  robustness_check.py     prompt-wording robustness sweep (variant_a / variant_b phrasing)
  parser.py                JSON extraction from raw model output
  schema.py                output row schema (TrialResult)
  output.py                CSV read/write helpers (resumable runs)

  analysis/               first-pass analysis pipeline, organised by research question
    shared/config.py       shared paths, labels, colour palette
    RQ1_baseline_heuristic/ RQ2_implicit_framing/ RQ3_explicit_symmetry/
    RQ4_architecture_equity/ robustness/ qualitative/

  new_analysis/           current analysis pipeline (sequential 00-18 scripts)
    config.py               shared paths, labels, colour palette
    00.._18..py             data completion -> model fitting -> effect sizes -> figures
    appendix*.tex            generated LaTeX tables for the manuscript appendix
    outputs/                 generated CSVs and figures (included)

outputs/new_experiment/  aggregated CSV outputs from the analysis/ pipeline (included)
```

## What's included vs. excluded

This is a code + aggregated-results export, not a full data dump:

- **Included:** all analysis and experiment-runner code, and every aggregated
  output the analysis pipelines produce (summary tables, model-fit
  coefficients, effect-size tables, figures, the raw per-trial model responses).


To reproduce from scratch:

1. Run `full_experiment.py` (and optionally `robustness_check.py`) against
   locally-hosted models to regenerate `results/new_experiment/*.csv`.
2. Run the scripts in `New_Experiment/analysis/` and/or
   `New_Experiment/new_analysis/` in numeric order to regenerate the tables
   and figures already provided under `outputs/` and `new_analysis/outputs/`.

## Requirements

See `requirements.txt`. Model inference expects either a local Ollama
installation or locally cached HuggingFace model weights (`llm.py` sets
`HF_HUB_OFFLINE=1` and reads from `HUGGINGFACE_HUB_CACHE`, default
`~/.cache/huggingface` — override via that environment variable if your
weights live elsewhere).

## License

CC BY-NC-SA 4.0, plus a kaitiakitanga statement asking that this work not be used to
justify deficit framings of Māori or other Indigenous communication norms.
See `LICENSE.md`.

## Note on AI-assisted code

Portions of this codebase were written with the assistance of an AI coding
tool. The experimental design, prompts, statistical methodology, and
interpretation of results are the author's own.
