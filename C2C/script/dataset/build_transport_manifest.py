"""Build a deterministic transport train/dev manifest from JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rosetta.transport.manifest import build_transport_manifest, save_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.01)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sample_ids = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if args.id_field not in record:
                raise ValueError(f"line {line_number} has no {args.id_field!r} field")
            sample_ids.append(record[args.id_field])
    manifest = build_transport_manifest(
        sample_ids, seed=args.seed, dev_fraction=args.dev_fraction
    )
    save_manifest(manifest, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
