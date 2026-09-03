"""Summarize versioned per-sample transport evaluation JSONL records."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rosetta.transport.ablation import paired_evaluation_summary
from rosetta.transport.evaluation import (
    latest_evaluation_records,
    save_evaluation_summary,
    summarize_evaluation_records,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reference-records",
        type=Path,
        help="Optional baseline JSONL for explicit sample-ID paired statistics",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = list(latest_evaluation_records(args.records).values())
    summary = summarize_evaluation_records(records)
    if args.reference_records is not None:
        reference = list(latest_evaluation_records(args.reference_records).values())
        summary["paired_comparison"] = paired_evaluation_summary(reference, records)
    save_evaluation_summary(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
