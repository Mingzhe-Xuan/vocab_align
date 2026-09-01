"""Build deterministic bidirectional byte-LSH candidates for vocabulary OT."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from rosetta.transport.ann_candidates import (
    ByteLshConfig,
    build_bidirectional_lsh_candidates,
)


def _git_version() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def save_candidates(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--min-ngram", type=int, default=1)
    parser.add_argument("--max-ngram", type=int, default=4)
    parser.add_argument("--signature-bits", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=128)
    parser.add_argument("--bridge-evidence", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-version")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    source = AutoTokenizer.from_pretrained(
        args.source, revision=args.source_revision, use_fast=True
    )
    target = AutoTokenizer.from_pretrained(
        args.target, revision=args.target_revision, use_fast=True
    )
    config = ByteLshConfig(
        dimension=args.dimension,
        min_ngram=args.min_ngram,
        max_ngram=args.max_ngram,
        signature_bits=args.signature_bits,
        top_k=args.top_k,
        pool_size=args.pool_size,
        bridge_evidence=args.bridge_evidence,
    )
    payload = build_bidirectional_lsh_candidates(
        source,
        target,
        config=config,
        seed=args.seed,
        code_version=args.code_version or _git_version(),
    )
    payload["requested_revisions"] = {
        "source": args.source_revision,
        "target": args.target_revision,
    }
    save_candidates(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
