"""Materialize a pinned deterministic corpus subset and its transport manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rosetta.transport.corpus_materialization import materialize_corpus


def _jsonl_factory(path: Path):
    def records() -> Iterable[Mapping]:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} must contain an object")
                yield value

    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--raw-split", default="train")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--records-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.01)
    parser.add_argument("--conversations-field", default="conversations")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_jsonl is not None:
        factory = _jsonl_factory(args.input_jsonl)
    else:
        from datasets import load_dataset

        dataset = load_dataset(
            args.dataset,
            revision=args.dataset_revision,
            split=args.raw_split,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        factory = lambda: iter(dataset)
    materialize_corpus(
        factory,
        args.records_output,
        args.manifest_output,
        dataset=args.dataset,
        revision=args.dataset_revision,
        raw_split=args.raw_split,
        sample_count=args.sample_count,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
        conversations_field=args.conversations_field,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
