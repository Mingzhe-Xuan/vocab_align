"""Freeze canonical messages, rendered prompts, and runtime provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

import yaml

from rosetta.transport.baseline import freeze_baseline, save_baseline
from rosetta.transport.config import TransportConfig


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--messages", required=True, type=Path)
    parser.add_argument("--rendered-prompts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-version")
    return parser.parse_args(argv)


def _git_version() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = TransportConfig.from_dict(config_payload)
    messages = json.loads(args.messages.read_text(encoding="utf-8"))
    prompts = json.loads(args.rendered_prompts.read_text(encoding="utf-8"))
    snapshot = freeze_baseline(
        config,
        messages,
        prompts,
        code_version=args.code_version or _git_version(),
    )
    save_baseline(snapshot, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
