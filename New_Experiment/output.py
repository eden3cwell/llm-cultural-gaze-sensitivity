"""
output.py — CSV read and write helpers.

"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from schema import CSV_FIELDS, TrialResult


def write_csv(rows: Iterable[TrialResult], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_row())
    return out_path


def append_csv(rows: Iterable[TrialResult], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r.to_row())
    return out_path


def load_completed(out_path: str | Path) -> set[tuple]:
    """
    Return set of (model, condition, fixation_rate, run_id, prompt_variant)
    tuples already written. Includes prompt_variant so robustness check rows
    are tracked independently of primary experiment rows.
    """
    out_path = Path(out_path)
    if not out_path.exists():
        return set()
    completed = set()
    with out_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                completed.add((
                    row["model"],
                    row["condition"],
                    int(row["fixation_rate"]),
                    int(row["run_id"]),
                    row.get("prompt_variant", "standard"),
                ))
            except (KeyError, ValueError):
                pass
    return completed
