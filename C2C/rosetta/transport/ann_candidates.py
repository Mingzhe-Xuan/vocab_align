"""Deterministic bidirectional LSH candidates over shared byte-ngram features."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .token_metadata import special_id_to_kind, token_raw_bytes, tokenizer_fingerprint


class AnnCandidateError(ValueError):
    """Raised when deterministic ANN candidate construction is invalid."""


@dataclass(frozen=True)
class ByteLshConfig:
    dimension: int = 256
    min_ngram: int = 1
    max_ngram: int = 4
    signature_bits: int = 16
    top_k: int = 8
    pool_size: int = 128
    bridge_evidence: float = 1e-6

    def validate(self) -> None:
        if self.dimension <= 0:
            raise AnnCandidateError("dimension must be positive")
        if (
            self.min_ngram <= 0
            or self.max_ngram < self.min_ngram
            or self.max_ngram > 255
        ):
            raise AnnCandidateError("ngram range is invalid")
        if not 1 <= self.signature_bits <= 32:
            raise AnnCandidateError("signature_bits must be in [1, 32]")
        if self.top_k <= 0 or self.pool_size < self.top_k:
            raise AnnCandidateError("pool_size must be at least positive top_k")
        if not np.isfinite(self.bridge_evidence) or not 0 < self.bridge_evidence < 1e-5:
            raise AnnCandidateError(
                "bridge_evidence must be finite, positive, and below ANN evidence"
            )


def _ordinary_ids(tokenizer: Any) -> Tuple[int, ...]:
    specials = set(special_id_to_kind(tokenizer))
    return tuple(
        sorted(
            {
                int(token_id)
                for token_id in tokenizer.get_vocab().values()
                if int(token_id) not in specials
            }
        )
    )


def _hashed_features(
    raw_values: Sequence[bytes], config: ByteLshConfig, seed: int
) -> np.ndarray:
    features = np.zeros((len(raw_values), config.dimension), dtype=np.float32)
    seed_bytes = int(seed).to_bytes(8, "little", signed=False)
    for row, raw in enumerate(raw_values):
        emitted = False
        for width in range(config.min_ngram, config.max_ngram + 1):
            if len(raw) < width:
                continue
            for start in range(len(raw) - width + 1):
                digest = hashlib.blake2b(
                    seed_bytes + bytes((width,)) + raw[start : start + width],
                    digest_size=8,
                ).digest()
                value = int.from_bytes(digest, "little")
                index = value % config.dimension
                features[row, index] += 1.0 if (value >> 32) & 1 else -1.0
                emitted = True
        if not emitted:
            digest = hashlib.blake2b(seed_bytes + raw, digest_size=8).digest()
            features[row, int.from_bytes(digest, "little") % config.dimension] = 1.0
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise AnnCandidateError("byte-ngram hashing produced a zero feature")
    return features / norms


def _signatures(features: np.ndarray, config: ByteLshConfig, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    projections = generator.standard_normal(
        (config.dimension, config.signature_bits), dtype=np.float32
    )
    bits = features @ projections >= 0
    powers = np.uint64(1) << np.arange(config.signature_bits, dtype=np.uint64)
    return (bits.astype(np.uint64) * powers).sum(axis=1)


def _indices_by_value(values: Iterable[int]) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = defaultdict(list)
    for index, value in enumerate(values):
        result[int(value)].append(index)
    return dict(result)


def _candidate_pool(
    signature: int,
    raw_length: int,
    signature_index: Mapping[int, Sequence[int]],
    length_index: Mapping[int, Sequence[int]],
    *,
    pool_size: int,
    maximum_length: int,
) -> List[int]:
    pool = list(signature_index.get(signature, ()))[:pool_size]
    if len(pool) >= pool_size:
        return pool
    seen = set(pool)
    for delta in range(maximum_length + 1):
        lengths = (
            (raw_length,) if delta == 0 else (raw_length - delta, raw_length + delta)
        )
        for length in lengths:
            if length < 0:
                continue
            for index in length_index.get(length, ()):
                if index not in seen:
                    seen.add(index)
                    pool.append(index)
                    if len(pool) >= pool_size:
                        return pool
    return pool


def _top_neighbors(
    query: np.ndarray,
    pool: Sequence[int],
    values: np.ndarray,
    value_ids: Sequence[int],
    top_k: int,
) -> List[Tuple[int, float]]:
    if not pool:
        raise AnnCandidateError("ANN candidate pool is empty")
    pool_array = np.asarray(pool, dtype=np.int64)
    scores = values[pool_array] @ query
    ranked = sorted(
        zip(pool_array.tolist(), scores.tolist()),
        key=lambda item: (-item[1], value_ids[item[0]]),
    )[:top_k]
    return [
        (int(value_ids[index]), max(1e-5, float((score + 1.0) / 2.0)))
        for index, score in ranked
    ]


def build_bidirectional_lsh_candidates(
    source_tokenizer: Any,
    target_tokenizer: Any,
    *,
    config: ByteLshConfig,
    seed: int,
    code_version: str,
) -> Dict[str, Any]:
    config.validate()
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= np.iinfo(np.uint64).max
    ):
        raise AnnCandidateError("seed must be an unsigned 64-bit integer")
    if not code_version.strip():
        raise AnnCandidateError("code_version is required")
    source_ids = _ordinary_ids(source_tokenizer)
    target_ids = _ordinary_ids(target_tokenizer)
    if not source_ids or not target_ids:
        raise AnnCandidateError("both tokenizers need ordinary tokens")
    source_raw = [
        token_raw_bytes(source_tokenizer, token_id) for token_id in source_ids
    ]
    target_raw = [
        token_raw_bytes(target_tokenizer, token_id) for token_id in target_ids
    ]
    source_features = _hashed_features(source_raw, config, seed)
    target_features = _hashed_features(target_raw, config, seed)
    source_signatures = _signatures(source_features, config, seed)
    target_signatures = _signatures(target_features, config, seed)
    source_signature_index = _indices_by_value(source_signatures.tolist())
    target_signature_index = _indices_by_value(target_signatures.tolist())
    source_length_index = _indices_by_value(len(value) for value in source_raw)
    target_length_index = _indices_by_value(len(value) for value in target_raw)
    max_length = max(max(map(len, source_raw)), max(map(len, target_raw)))
    candidates: Dict[int, Dict[int, float]] = {
        source_id: {} for source_id in source_ids
    }

    for index, source_id in enumerate(source_ids):
        pool = _candidate_pool(
            int(source_signatures[index]),
            len(source_raw[index]),
            target_signature_index,
            target_length_index,
            pool_size=config.pool_size,
            maximum_length=max_length,
        )
        for target_id, evidence in _top_neighbors(
            source_features[index], pool, target_features, target_ids, config.top_k
        ):
            candidates[source_id][target_id] = max(
                candidates[source_id].get(target_id, 0.0), evidence
            )

    for index, target_id in enumerate(target_ids):
        pool = _candidate_pool(
            int(target_signatures[index]),
            len(target_raw[index]),
            source_signature_index,
            source_length_index,
            pool_size=config.pool_size,
            maximum_length=max_length,
        )
        for source_id, evidence in _top_neighbors(
            target_features[index], pool, source_features, source_ids, config.top_k
        ):
            candidates[source_id][target_id] = max(
                candidates[source_id].get(target_id, 0.0), evidence
            )

    source_anchor = source_ids[0]
    target_anchor = target_ids[0]
    for source_id in source_ids:
        candidates[source_id].setdefault(target_anchor, config.bridge_evidence)
    for target_id in target_ids:
        candidates[source_anchor].setdefault(target_id, config.bridge_evidence)

    source_fingerprint = tokenizer_fingerprint(source_tokenizer)
    target_fingerprint = tokenizer_fingerprint(target_tokenizer)
    fingerprint_payload = {
        "source_fingerprint": source_fingerprint,
        "target_fingerprint": target_fingerprint,
        "build_config": asdict(config),
        "seed": seed,
    }
    input_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    serialized = {
        str(source_id): [
            [target_id, evidence] for target_id, evidence in sorted(targets.items())
        ]
        for source_id, targets in sorted(candidates.items())
    }
    return {
        "schema_version": 1,
        "input_fingerprint": input_fingerprint,
        "source_fingerprint": source_fingerprint,
        "target_fingerprint": target_fingerprint,
        "build_config": {
            "method": "seeded-hashed-byte-ngram-lsh-bidirectional",
            **asdict(config),
        },
        "seed": seed,
        "code_version": code_version,
        "coverage": {
            "ordinary_source_tokens": len(source_ids),
            "ordinary_target_tokens": len(target_ids),
            "source_tokens_with_candidates": len(serialized),
            "target_tokens_with_candidates": len(
                {target_id for targets in candidates.values() for target_id in targets}
            ),
            "source_anchor": source_anchor,
            "target_anchor": target_anchor,
        },
        "candidates": serialized,
    }
