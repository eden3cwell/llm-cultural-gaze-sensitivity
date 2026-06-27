#!/usr/bin/env python3
"""
full_experiment.py — parametric sweep for the focused cultural gaze study.

"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

from prompts import build_messages
from design import (
    MODELS, CONDITIONS, FIXATION_RATES,
    N_RUNS, TEMPERATURE, TOP_P, MAX_NEW_TOKENS, BASE_SEED, DIFFICULTY_LEVEL,
)
from llm import LocalLLM
from parser import parse_into_result
from output import append_csv, load_completed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models",           nargs="+",  default=MODELS)
    ap.add_argument("--conditions",       nargs="+",  default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument("--output",           default="results/new_experiment.csv")
    ap.add_argument("--runs",             type=int,   default=N_RUNS)
    ap.add_argument("--temperature",      type=float, default=TEMPERATURE)
    ap.add_argument("--top-p",            type=float, default=TOP_P)
    ap.add_argument("--max-new-tokens",   type=int,   default=MAX_NEW_TOKENS)
    ap.add_argument("--base-seed",        type=int,   default=BASE_SEED)
    ap.add_argument("--difficulty-level", type=float, default=DIFFICULTY_LEVEL)
    args = ap.parse_args()

    out_path  = Path(args.output)
    completed = load_completed(out_path)

    total = len(args.models) * len(args.conditions) * len(FIXATION_RATES) * args.runs
    done  = len(completed)

    print(f"[experiment] models         = {len(args.models)}",      flush=True)
    print(f"[experiment] conditions     = {args.conditions}",        flush=True)
    print(f"[experiment] fixation rates = {len(FIXATION_RATES)}",   flush=True)
    print(f"[experiment] runs/cell      = {args.runs}",              flush=True)
    print(f"[experiment] temperature    = {args.temperature}",       flush=True)
    print(f"[experiment] top_p          = {args.top_p}",             flush=True)
    print(f"[experiment] difficulty     = {args.difficulty_level}",  flush=True)
    print(f"[experiment] total trials   = {total}",                  flush=True)
    print(f"[experiment] already done   = {done}",                   flush=True)
    print(f"[experiment] remaining      = {total - done}",           flush=True)
    print(f"[experiment] output         = {out_path}",               flush=True)
    print("-" * 72, flush=True)

    overall  = done
    n_failed = 0

    for model_id in args.models:
        pending = [
            (condition, rate, run_id)
            for condition in args.conditions
            for rate      in FIXATION_RATES
            for run_id    in range(args.runs)
            if (model_id, condition, rate, run_id, "standard") not in completed
        ]

        if not pending:
            print(f"[experiment] SKIP {model_id} — all conditions complete.",
                  flush=True)
            continue

        print(f"\n[experiment] Loading: {model_id}",           flush=True)
        print(f"[experiment] Pending: {len(pending)} trials",  flush=True)

        try:
            llm = LocalLLM(
                model_name=model_id,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                seed=args.base_seed,
            )
        except Exception as e:
            print(f"[experiment] FATAL: could not load {model_id}: {e}", flush=True)
            n_failed += len(pending)
            continue

        for condition, rate, run_id in pending:
            overall += 1

            if hasattr(llm, "_torch"):
                llm._torch.manual_seed(args.base_seed + run_id)

            print(
                f"  [{overall}/{total}] {model_id} | "
                f"condition={condition} | fixation={rate}% | run={run_id}",
                flush=True,
            )

            try:
                messages = build_messages(
                    rate,
                    difficulty_level=args.difficulty_level,
                    condition=condition,
                    prompt_variant="standard",
                )
                raw    = llm.generate(messages)
                result = parse_into_result(
                    raw_output=raw,
                    fixation_rate=rate,
                    model=model_id,
                    run_id=run_id,
                    difficulty_level=args.difficulty_level,
                    condition=condition,
                    prompt_variant="standard",
                )
            except Exception as e:
                print(f"    ERROR: {e}", flush=True)
                n_failed += 1
                continue

            append_csv([result], out_path)

            if result.parse_error:
                print(f"    PARSE ERROR: {result.parse_error}", flush=True)
                n_failed += 1
            else:
                delta = result.difficulty_delta
                delta_str = (
                    f"{delta:+.2f}" if isinstance(delta, (int, float)) else str(delta)
                )
                print(
                    f"    engagement={result.engagement_score} "
                    f"direction={result.delta_direction} "
                    f"delta={delta_str} "
                    f"label={result.adaptation_label} ({result.adaptation_label_num}) "
                    f"confidence={result.confidence}",
                    flush=True,
                )

        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[experiment] {model_id} complete. GPU cache cleared.", flush=True)

    print("\n" + "=" * 72,                                           flush=True)
    print(f"[experiment] Finished. {overall}/{total} trials attempted.", flush=True)
    print(f"[experiment] Parse/load failures: {n_failed}",              flush=True)
    print(f"[experiment] Results written to:  {out_path}",              flush=True)

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
