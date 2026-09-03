"""Expand a pre-registered transport ablation plan without running models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import yaml

from rosetta.transport.ablation import AblationPlan


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("dev", "test"))
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = AblationPlan.from_dict(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    runs = plan.expand(args.phase)
    payload = {
        "schema_version": 1,
        "phase": args.phase,
        "run_count": len(runs),
        "runs": [run.to_dict() for run in runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
