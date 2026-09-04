"""Create a Bayes-reversed sparse vocabulary transport artifact."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Sequence

from rosetta.transport.artifact import (
    load_transport_artifact,
    save_transport_artifact,
)
from rosetta.transport.audit import audit_transport_artifact, save_audit
from rosetta.transport.reversal import reverse_transport_artifact


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--audit-markdown", required=True, type=Path)
    parser.add_argument("--code-version")
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = args.artifact.resolve()
    if input_path == output_path:
        parser.error("--input and --artifact must be different paths")
    if args.artifact.exists():
        parser.error("refusing to overwrite an existing artifact")

    artifact = load_transport_artifact(args.input)
    reversed_artifact = reverse_transport_artifact(
        artifact,
        code_version=args.code_version or _git_version(),
        parent_sha256=_file_sha256(args.input),
    )
    partial = args.artifact.with_suffix(args.artifact.suffix + ".partial.npz")
    if partial.exists():
        parser.error(f"refusing to overwrite partial artifact: {partial}")
    save_transport_artifact(reversed_artifact, partial)
    verified = load_transport_artifact(partial)
    report = audit_transport_artifact(verified)
    if not report["valid"]:
        raise ValueError("reversed artifact failed independent audit")
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(args.artifact)
    save_audit(report, args.audit_json, args.audit_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
