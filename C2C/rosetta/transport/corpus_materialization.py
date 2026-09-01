"""Deterministic materialization of a pinned source-dataset prefix."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .corpus import build_corpus_manifest, canonical_messages
from .config import PINNED_REVISION
from .manifest import serialize_manifest


class MaterializationError(ValueError):
    """Raised when a pinned corpus cannot be materialized reproducibly."""


RecordFactory = Callable[[], Iterable[Mapping[str, Any]]]


def materialize_corpus(
    record_factory: RecordFactory,
    records_output: str | Path,
    manifest_output: str | Path,
    *,
    dataset: str,
    revision: str,
    raw_split: str = "train",
    sample_count: int = 500_000,
    seed: int = 42,
    dev_fraction: float = 0.01,
    conversations_field: str = "conversations",
) -> dict[str, Any]:
    """Save the first ``sample_count`` source rows and their canonical manifest.

    Taking the pinned split prefix deliberately mirrors
    ``OpenHermesChatDataset.select(range(num_samples))``. The seed controls the
    subsequent train/dev split; it does not silently select a different corpus.
    """
    if not isinstance(dataset, str) or not dataset.strip():
        raise MaterializationError("dataset is required")
    if not isinstance(revision, str) or not PINNED_REVISION.fullmatch(revision):
        raise MaterializationError(
            "dataset revision must be a pinned 40-character commit SHA"
        )
    if (
        not isinstance(raw_split, str)
        or not raw_split.strip()
        or "test" in raw_split.lower()
    ):
        raise MaterializationError("raw split must be non-test")
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise MaterializationError("sample count must be a positive integer")

    records_output = Path(records_output)
    manifest_output = Path(manifest_output)
    if records_output.resolve() == manifest_output.resolve():
        raise MaterializationError("records and manifest outputs must differ")
    records_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    records_partial = records_output.with_name(records_output.name + ".partial")
    manifest_partial = manifest_output.with_name(manifest_output.name + ".partial")
    written = 0
    try:
        with records_partial.open("w", encoding="utf-8", newline="\n") as handle:
            for index, record in enumerate(record_factory()):
                if index >= sample_count:
                    break
                if not isinstance(record, Mapping):
                    raise MaterializationError(f"record {index} must be an object")
                canonical_messages(record, conversations_field=conversations_field)
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                written += 1
        if written != sample_count:
            raise MaterializationError(
                f"materialized {written} source rows, expected {sample_count}"
            )
        manifest = build_corpus_manifest(
            records_partial,
            dataset=dataset,
            revision=revision,
            raw_split=raw_split,
            conversations_field=conversations_field,
            seed=seed,
            dev_fraction=dev_fraction,
        )
        manifest["selection"] = {
            "algorithm": "pinned-source-prefix-v1",
            "source_start_index": 0,
            "requested_count": sample_count,
            "selected_source_rows": written,
            "unique_conversations": manifest["unique_record_count"],
            "adapter_filtering": "not-applied",
            "split_seed": seed,
        }
        manifest_partial.write_bytes(serialize_manifest(manifest))
        os.replace(records_partial, records_output)
        os.replace(manifest_partial, manifest_output)
        return manifest
    except Exception:
        records_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise
