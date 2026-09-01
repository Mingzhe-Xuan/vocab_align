"""Small, auditable vocabulary alignment based on bytes and text spans.

This is deliberately a local column-normalized baseline.  It does not use
ANN fallback or optimal transport, and it only creates columns for source
tokens observed in the supplied texts (plus observed special tokens).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple


def _bytes_to_unicode() -> Dict[int, str]:
    """Return the reversible byte alphabet used by GPT/Qwen byte-level BPE."""
    safe = list(range(ord("!"), ord("~") + 1))
    safe += list(range(ord("¡"), ord("¬") + 1))
    safe += list(range(ord("®"), ord("ÿ") + 1))
    chars = safe[:]
    extra = 0
    for byte in range(256):
        if byte not in safe:
            safe.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(safe, (chr(value) for value in chars)))


_UNICODE_TO_BYTE = {value: key for key, value in _bytes_to_unicode().items()}


def token_raw_bytes(tokenizer: Any, token_id: int) -> bytes:
    """Recover raw bytes for a normal byte-level BPE token.

    Special tokens are represented by their literal UTF-8 spelling and are
    handled separately by the alignment priority rules.
    """
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        raise ValueError(f"Tokenizer returned no token for id {token_id}")
    if all(char in _UNICODE_TO_BYTE for char in token):
        return bytes(_UNICODE_TO_BYTE[char] for char in token)
    return token.encode("utf-8")


def _special_id_to_token(tokenizer: Any) -> Dict[int, str]:
    vocabulary = tokenizer.get_vocab()
    result: Dict[int, str] = {}
    for token in getattr(tokenizer, "all_special_tokens", []):
        token_id = vocabulary.get(token)
        if token_id is not None:
            result[int(token_id)] = token
    return result


def _encode_with_byte_spans(tokenizer: Any, text: str) -> List[Tuple[int, int, int]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]

    byte_prefix = [0]
    for char in text:
        byte_prefix.append(byte_prefix[-1] + len(char.encode("utf-8")))

    spans: List[Tuple[int, int, int]] = []
    for token_id, (char_start, char_end) in zip(ids, offsets):
        if not (0 <= char_start <= char_end <= len(text)):
            raise ValueError(
                f"Invalid character offset {(char_start, char_end)} for text length {len(text)}"
            )
        if char_start == char_end:
            continue
        spans.append((int(token_id), byte_prefix[char_start], byte_prefix[char_end]))
    return spans


def _tokenizer_fingerprint(tokenizer: Any) -> str:
    payload = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
        "vocab": sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]),
        "special_tokens": list(getattr(tokenizer, "all_special_tokens", [])),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class SparseColumn:
    source_id: int
    source_token: str
    source_bytes_hex: str
    source_count: int
    rule: str
    target_ids: List[int]
    weights: List[float]


@dataclass
class LocalTransportArtifact:
    schema_version: int
    source_tokenizer: str
    target_tokenizer: str
    source_fingerprint: str
    target_fingerprint: str
    text_count: int
    observed_source_tokens: int
    observed_target_tokens: int
    columns: List[SparseColumn]
    audit: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_small_transport(
    source_tokenizer: Any,
    target_tokenizer: Any,
    texts: Iterable[str],
) -> LocalTransportArtifact:
    """Build special/exact/span columns for tokens observed in ``texts``."""
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError("At least one non-empty alignment text is required")
    if not getattr(source_tokenizer, "is_fast", False):
        raise ValueError("source_tokenizer must be a fast tokenizer with offsets")
    if not getattr(target_tokenizer, "is_fast", False):
        raise ValueError("target_tokenizer must be a fast tokenizer with offsets")

    source_special = _special_id_to_token(source_tokenizer)
    target_special_by_token = {
        token: token_id for token_id, token in _special_id_to_token(target_tokenizer).items()
    }
    target_special_ids = set(_special_id_to_token(target_tokenizer))
    target_bytes: DefaultDict[bytes, List[int]] = defaultdict(list)
    for target_id in range(len(target_tokenizer)):
        if target_id not in target_special_ids:
            target_bytes[token_raw_bytes(target_tokenizer, target_id)].append(target_id)

    source_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    span_counts: DefaultDict[int, Counter[int]] = defaultdict(Counter)

    for text in texts:
        source_spans = _encode_with_byte_spans(source_tokenizer, text)
        target_spans = _encode_with_byte_spans(target_tokenizer, text)
        source_counts.update(token_id for token_id, _, _ in source_spans)
        target_counts.update(token_id for token_id, _, _ in target_spans)

        target_cursor = 0
        for source_id, source_start, source_end in source_spans:
            while target_cursor < len(target_spans) and target_spans[target_cursor][2] <= source_start:
                target_cursor += 1
            cursor = target_cursor
            while cursor < len(target_spans) and target_spans[cursor][1] < source_end:
                target_id, target_start, target_end = target_spans[cursor]
                overlap = min(source_end, target_end) - max(source_start, target_start)
                if overlap > 0:
                    span_counts[source_id][target_id] += overlap
                cursor += 1

    columns: List[SparseColumn] = []
    rule_counts: Counter[str] = Counter()
    duplicate_exact_columns = 0
    for source_id in sorted(source_counts):
        source_token = source_tokenizer.convert_ids_to_tokens(source_id)
        raw_bytes = token_raw_bytes(source_tokenizer, source_id)
        rule: str
        candidates: Mapping[int, int | float]

        if source_id in source_special and source_special[source_id] in target_special_by_token:
            rule = "special"
            candidates = {target_special_by_token[source_special[source_id]]: 1.0}
        else:
            exact_ids = target_bytes.get(raw_bytes, [])
            if exact_ids:
                rule = "exact_byte"
                # Identical byte strings should normally be unique.  A uniform
                # column preserves determinism without silently picking an ID.
                candidates = {target_id: 1.0 for target_id in exact_ids}
                duplicate_exact_columns += int(len(exact_ids) > 1)
            elif span_counts[source_id]:
                rule = "byte_span"
                candidates = span_counts[source_id]
            else:
                rule = "uncovered"
                candidates = {}

        total = float(sum(candidates.values()))
        target_ids = sorted(candidates)
        weights = [float(candidates[target_id]) / total for target_id in target_ids] if total else []
        rule_counts[rule] += 1
        columns.append(
            SparseColumn(
                source_id=source_id,
                source_token=str(source_token),
                source_bytes_hex=raw_bytes.hex(),
                source_count=source_counts[source_id],
                rule=rule,
                target_ids=target_ids,
                weights=weights,
            )
        )

    covered_occurrences = sum(
        column.source_count for column in columns if column.rule != "uncovered"
    )
    total_occurrences = sum(source_counts.values())
    max_column_error = max(
        (abs(sum(column.weights) - 1.0) for column in columns if column.weights),
        default=0.0,
    )
    audit = {
        "rule_column_counts": dict(sorted(rule_counts.items())),
        "covered_column_fraction": (
            sum(column.rule != "uncovered" for column in columns) / len(columns)
        ),
        "covered_occurrence_fraction": covered_occurrences / total_occurrences,
        "duplicate_exact_columns": duplicate_exact_columns,
        "max_column_sum_error": max_column_error,
        "nonnegative": all(weight >= 0 for column in columns for weight in column.weights),
    }
    return LocalTransportArtifact(
        schema_version=1,
        source_tokenizer=str(getattr(source_tokenizer, "name_or_path", "unknown")),
        target_tokenizer=str(getattr(target_tokenizer, "name_or_path", "unknown")),
        source_fingerprint=_tokenizer_fingerprint(source_tokenizer),
        target_fingerprint=_tokenizer_fingerprint(target_tokenizer),
        text_count=len(texts),
        observed_source_tokens=len(source_counts),
        observed_target_tokens=len(target_counts),
        columns=columns,
        audit=audit,
    )


def save_transport(artifact: LocalTransportArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_transport(path: str | Path) -> LocalTransportArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["columns"] = [SparseColumn(**column) for column in payload["columns"]]
    return LocalTransportArtifact(**payload)
