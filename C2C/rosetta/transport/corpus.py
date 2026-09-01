"""Manifest-bound canonical conversation corpora for transport fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .config import PINNED_REVISION
from .manifest import build_transport_manifest


class CorpusError(ValueError):
    """Raised when raw records cannot reproduce a manifest-bound corpus."""


_ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
}


def canonical_messages(
    record: Mapping[str, Any], *, conversations_field: str = "conversations"
) -> Tuple[Dict[str, str], ...]:
    conversations = record.get(conversations_field)
    if not isinstance(conversations, list) or not conversations:
        raise CorpusError("record conversations must be a non-empty list")
    messages = []
    for index, message in enumerate(conversations):
        if not isinstance(message, Mapping):
            raise CorpusError(f"conversation message {index} must be an object")
        role = message.get("from", message.get("role"))
        text = message.get("value", message.get("content"))
        if not isinstance(role, str) or role not in _ROLE_MAP:
            raise CorpusError(f"unsupported conversation role at message {index}")
        if not isinstance(text, str) or not text.strip():
            raise CorpusError(f"conversation text {index} must be non-empty")
        messages.append({"role": _ROLE_MAP[str(role)], "text": text})
    return tuple(messages)


def canonical_sample_id(messages: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(
        list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "conversation-sha256:" + hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_records(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CorpusError(f"invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise CorpusError(f"line {line_number} must contain an object")
            yield line_number, record


def build_corpus_manifest(
    records_path: str | Path,
    *,
    dataset: str,
    revision: str,
    raw_split: str = "train",
    conversations_field: str = "conversations",
    seed: int = 42,
    dev_fraction: float = 0.01,
) -> Dict[str, Any]:
    path = Path(records_path)
    if not dataset.strip():
        raise CorpusError("dataset is required")
    if not PINNED_REVISION.fullmatch(revision):
        raise CorpusError("dataset revision must be a pinned 40-character commit SHA")
    if not raw_split.strip() or "test" in raw_split.lower():
        raise CorpusError("raw split must be non-test")
    sample_ids = []
    seen = set()
    duplicate_count = 0
    record_count = 0
    for _, record in _iter_records(path):
        record_count += 1
        sample_id = canonical_sample_id(
            canonical_messages(record, conversations_field=conversations_field)
        )
        if sample_id in seen:
            duplicate_count += 1
            continue
        seen.add(sample_id)
        sample_ids.append(sample_id)
    manifest = build_transport_manifest(
        sample_ids, seed=seed, dev_fraction=dev_fraction
    )
    manifest.update(
        {
            "dataset": dataset,
            "dataset_revision": revision,
            "raw_split": raw_split,
            "identity_scheme": "canonical-conversation-sha256-v1",
            "conversations_field": conversations_field,
            "raw_input_sha256": file_sha256(path),
            "raw_record_count": record_count,
            "unique_record_count": len(sample_ids),
            "duplicate_content_records": duplicate_count,
        }
    )
    return manifest


def load_manifest_texts(
    records_path: str | Path,
    manifest_path: str | Path,
    *,
    build_split: str,
) -> Tuple[List[str], Dict[str, Any]]:
    if build_split not in {"transport_train", "transport_dev"}:
        raise CorpusError("build split must be transport_train or transport_dev")
    records_path = Path(records_path)
    manifest_path = Path(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise CorpusError("invalid corpus manifest JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise CorpusError("unsupported corpus manifest schema")
    required = {
        "dataset",
        "dataset_revision",
        "raw_split",
        "identity_scheme",
        "conversations_field",
        "raw_input_sha256",
        "input_fingerprint",
        "seed",
        "algorithm",
        "dev_fraction",
        "sample_count",
        "transport_train",
        "transport_dev",
    }
    missing = required.difference(manifest)
    if missing:
        raise CorpusError(f"corpus manifest provenance missing: {sorted(missing)}")
    if manifest["identity_scheme"] != "canonical-conversation-sha256-v1":
        raise CorpusError("unsupported corpus identity scheme")
    if not PINNED_REVISION.fullmatch(str(manifest["dataset_revision"])):
        raise CorpusError("corpus manifest dataset revision is not pinned")
    if "test" in str(manifest["raw_split"]).lower():
        raise CorpusError("corpus manifest raw split must be non-test")
    raw_hash = manifest["raw_input_sha256"]
    if (
        not isinstance(raw_hash, str)
        or len(raw_hash) != 64
        or any(character not in "0123456789abcdef" for character in raw_hash)
    ):
        raise CorpusError("corpus manifest raw SHA-256 is invalid")
    if file_sha256(records_path) != manifest["raw_input_sha256"]:
        raise CorpusError("raw corpus SHA-256 does not match manifest")
    train = manifest["transport_train"]
    dev = manifest["transport_dev"]
    if not isinstance(train, list) or not isinstance(dev, list):
        raise CorpusError("manifest transport splits must be lists")
    for values in (train, dev):
        if any(not isinstance(value, str) or not value for value in values) or len(
            values
        ) != len(set(values)):
            raise CorpusError("manifest splits must contain unique sample IDs")
    if set(train).intersection(dev):
        raise CorpusError("manifest transport splits overlap")
    conversations_field = manifest["conversations_field"]
    if not isinstance(conversations_field, str) or not conversations_field:
        raise CorpusError("manifest conversations field is invalid")
    unique_records = []
    raw_ids = []
    raw_seen = set()
    for _, record in _iter_records(records_path):
        messages = canonical_messages(record, conversations_field=conversations_field)
        sample_id = canonical_sample_id(messages)
        if sample_id in raw_seen:
            continue
        raw_seen.add(sample_id)
        raw_ids.append(sample_id)
        unique_records.append((sample_id, messages))
    try:
        reproduced = build_transport_manifest(
            raw_ids,
            seed=manifest["seed"],
            dev_fraction=manifest["dev_fraction"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CorpusError("corpus manifest split configuration is invalid") from exc
    split_keys = (
        "algorithm",
        "input_fingerprint",
        "sample_count",
        "transport_train",
        "transport_dev",
    )
    if any(manifest[key] != reproduced[key] for key in split_keys):
        raise CorpusError("corpus manifest split does not reproduce raw samples")
    wanted = set(manifest[build_split])
    texts: List[str] = []
    for sample_id, messages in unique_records:
        if sample_id in wanted:
            texts.extend(message["text"] for message in messages)
    if not texts:
        raise CorpusError("manifest split produced no canonical text")
    provenance = {
        "mode": "manifest-bound-canonical-conversations",
        "dataset": manifest["dataset"],
        "dataset_revision": manifest["dataset_revision"],
        "raw_split": manifest["raw_split"],
        "build_split": build_split,
        "identity_scheme": manifest["identity_scheme"],
        "raw_input_sha256": manifest["raw_input_sha256"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "selected_samples": len(wanted),
        "canonical_messages": len(texts),
    }
    return texts, provenance
