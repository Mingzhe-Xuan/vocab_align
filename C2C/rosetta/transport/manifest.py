"""Deterministic transport train/dev split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class ManifestError(ValueError):
    """Raised when stable sample IDs cannot form a valid manifest."""


def _fingerprint(sample_ids: List[str]) -> str:
    encoded = json.dumps(sample_ids, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_transport_manifest(
    sample_ids: Iterable[str], *, seed: int = 42, dev_fraction: float = 0.01
) -> Dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ManifestError("seed must be a nonnegative integer")
    if not 0 < dev_fraction < 1:
        raise ManifestError("dev_fraction must be between zero and one")
    values = list(sample_ids)
    if not values:
        raise ManifestError("at least one sample ID is required")
    if any(not isinstance(value, str) or not value for value in values):
        raise ManifestError("sample IDs must be non-empty strings")
    if len(values) != len(set(values)):
        raise ManifestError("duplicate sample IDs are not allowed")

    ordered = sorted(values)
    ranked = sorted(
        ordered,
        key=lambda value: (
            hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest(),
            value,
        ),
    )
    if len(ranked) == 1:
        dev_count = 0
    else:
        dev_count = min(len(ranked) - 1, max(1, round(len(ranked) * dev_fraction)))
    dev_set = set(ranked[:dev_count])
    train = [value for value in ordered if value not in dev_set]
    dev = [value for value in ordered if value in dev_set]
    return {
        "schema_version": 1,
        "seed": seed,
        "algorithm": "sha256-rank-v1",
        "dev_fraction": dev_fraction,
        "input_fingerprint": _fingerprint(ordered),
        "sample_count": len(ordered),
        "transport_train": train,
        "transport_dev": dev,
    }


def serialize_manifest(manifest: Dict[str, Any]) -> bytes:
    return (
        json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def save_manifest(manifest: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialize_manifest(manifest))
