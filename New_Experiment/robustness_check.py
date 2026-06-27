#!/usr/bin/env python3
"""
robustness_check.py — prompt sensitivity analysis.

"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

from prompts import build_messages
from design import (
    ROBUSTNESS_MODELS, ROBUSTNESS_VARIANTS, FIXATION_RATES,
    N_RUNS, TEMPERATURE, TOP_P, MAX_NEW_TOKENS, BASE_SEED, DIFFICULTY_LEVEL,
)
from llm import LocalLLM
from parser import parse_into_result
from output import append_csv, load_completed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models",           nargs="+",  default=ROBUSTNESS_MODELS)
    ap.add_argument("--variants",         nargs="+",  default=ROBUSTNESS_VARIANTS,
                    choices=ROBUSTNESS_VARIANTS)
    ap.add_argument("--output",           default="results/robustness_check.csv")
    ap.add_argument("--runs",             type=int,   default=N_RUNS)
    ap.add_argument("--temperature",      type=float, default=TEMPERATURE)
    ap.add_argument("--top-p",            type=float, default=TOP_P)
    ap.add_argument("--max-new-tokens",   type=int,   default=MAX_NEW_TOKENS)
    ap.add_argument("--base-seed",        type=int,   default=BASE_SEED)
    ap.add_argument("--difficulty-level", type=float, default=DIFFICULTY_LEVEL)
    args = ap.parse_args()

    out_path  = Path(args.output)
    completed = load_completed(out_path)

    total = len(args.models) * len(args.variants) * len(FIXATION_RATES) * args.runs
    done  = len(completed)

    print(f"[robustness] models         = {args.models}",           flush=True)
    print(f"[robustness] variants       = {args.variants}",         flush=True)
    print(f"[robustness] condition      = baseline (fixed)",         flush=True)
    print(f"[robustness] fixation rates = {len(FIXATION_RATES)}",   flush=True)
    print(f"[robustness] runs/cell      = {args.runs}",              flush=True)
    print(f"[robustness] temperature    = {args.temperature}",       flush=True)
    print(f"[robustness] top_p          = {args.top_p}",             flush=True)
    print(f"[robustness] total trials   = {total}",                  flush=True)
    print(f"[robustness] already done   = {done}",                   flush=True)
    print(f"[robustness] remaining      = {total - done}",           flush=True)
    print(f"[robustness] output         = {out_path}",               flush=True)
    print("-" * 72, flush=True)

    overall  = done
    n_failed = 0

    for model_id in args.models:
        pending = [
            (variant, rate, run_id)
            for variant in args.variants
            for rate    in FIXATION_RATES
            for run_id  in range(args.runs)
            if (model_id, "baseline", rate, run_id, variant) not in completed
        ]

        if not pending:
            print(f"[robustness] SKIP {model_id} — all variants complete.",
                  flush=True)
            continue

        print(f"\n[robustness] Loading: {model_id}",          flush=True)
        print(f"[robustness] Pending: {len(pending)} trials", flush=True)

        try:
            llm = LocalLLM(
                model_name=model_id,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                top_p=args.top_p,
                seed=args.base_seed,
            )
        except Exception as e:
            print(f"[robustness] FATAL: could not load {model_id}: {e}", flush=True)
            n_failed += len(pending)
            continue

        for variant, rate, run_id in pending:
            overall += 1

            if hasattr(llm, "_torch"):
                llm._torch.manual_seed(args.base_seed + run_id)

            print(
                f"  [{overall}/{total}] {model_id} | "
                f"variant={variant} | fixation={rate}% | run={run_id}",
                flush=True,
            )

            try:
                messages = build_messages(
                    rate,
                    difficulty_level=args.difficulty_level,
                    condition="baseline",
                    prompt_variant=variant,
                )
                raw    = llm.generate(messages)
                result = parse_into_result(
                    raw_output=raw,
                    fixation_rate=rate,
                    model=model_id,
                    run_id=run_id,
                    difficulty_level=args.difficulty_level,
                    condition="baseline",
                    prompt_variant=variant,
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
                    f"label={result.adaptation_label} "
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
        print(f"[robustness] {model_id} complete. GPU cache cleared.", flush=True)

    print("\n" + "=" * 72,                                               flush=True)
    print(f"[robustness] Finished. {overall}/{total} trials attempted.", flush=True)
    print(f"[robustness] Parse/load failures: {n_failed}",               flush=True)
    print(f"[robustness] Results written to:  {out_path}",               flush=True)

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
