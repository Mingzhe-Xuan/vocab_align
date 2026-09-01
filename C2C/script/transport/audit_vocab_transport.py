"""Load and independently audit a vocabulary transport artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rosetta.transport.artifact import load_transport_artifact
from rosetta.transport.audit import audit_transport_artifact, save_audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    parser.add_argument("--source-fingerprint")
    parser.add_argument("--target-fingerprint")
    args = parser.parse_args(argv)
    artifact = load_transport_artifact(
        args.artifact,
        source_fingerprint=args.source_fingerprint,
        target_fingerprint=args.target_fingerprint,
    )
    report = audit_transport_artifact(artifact)
    save_audit(report, args.json_output, args.markdown_output)
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
