"""Streaming token-frequency marginals over canonical message content."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Tuple

import numpy as np

from .token_metadata import encode_with_byte_spans, special_id_to_kind


class MarginalError(ValueError):
    """Raised when a valid positive active-support marginal cannot be formed."""


@dataclass(frozen=True)
class TokenMarginal:
    probabilities: np.ndarray
    counts: np.ndarray
    active_ids: Tuple[int, ...]

    def validate(self) -> None:
        active = self.probabilities[list(self.active_ids)]
        if not len(active) or np.any(active <= 0) or not np.all(np.isfinite(active)):
            raise MarginalError(
                "active marginal probabilities must be finite and positive"
            )
        if not np.isclose(self.probabilities.sum(), 1.0):
            raise MarginalError("marginal must sum to one")
        inactive = np.ones(len(self.probabilities), dtype=bool)
        inactive[list(self.active_ids)] = False
        if np.any(self.probabilities[inactive] != 0):
            raise MarginalError("inactive tokens must retain zero probability")


def estimate_token_marginal(
    tokenizer: Any,
    texts: Iterable[str],
    *,
    smoothing: float = 0.0,
    special_pseudocounts: Mapping[str, float] | None = None,
    excluded_special_kinds: Iterable[str] = ("pad", "unk", "mask"),
    allowed_token_ids: Iterable[int] | None = None,
) -> TokenMarginal:
    if not np.isfinite(smoothing) or smoothing < 0:
        raise MarginalError("smoothing must be finite and nonnegative")
    vocabulary_ids = {int(value) for value in tokenizer.get_vocab().values()}
    if allowed_token_ids is None:
        allowed = set(vocabulary_ids)
    else:
        raw_allowed = tuple(allowed_token_ids)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_allowed
        ):
            raise MarginalError("allowed token IDs must be integers")
        allowed = set(raw_allowed)
        if not allowed.issubset(vocabulary_ids):
            raise MarginalError("allowed token ID is outside tokenizer vocabulary")
    counts: Counter[int] = Counter()
    for text in texts:
        if text:
            counts.update(
                token_id for token_id, _, _ in encode_with_byte_spans(tokenizer, text)
            )
    kinds = special_id_to_kind(tokenizer)
    excluded = set(excluded_special_kinds)
    excluded_ids = {token_id for token_id, kind in kinds.items() if kind in excluded}
    for token_id, kind in kinds.items():
        if kind in excluded or token_id not in allowed:
            counts.pop(token_id, None)
    for token_id in tuple(counts):
        if token_id not in allowed:
            counts.pop(token_id)
    for kind, value in (special_pseudocounts or {}).items():
        if not np.isfinite(value) or value < 0:
            raise MarginalError("special pseudocounts must be finite and nonnegative")
        matches = [
            token_id for token_id, token_kind in kinds.items() if token_kind == kind
        ]
        if len(matches) != 1:
            raise MarginalError(f"special kind {kind!r} is not unambiguous")
        if matches[0] not in allowed or kind in excluded:
            raise MarginalError(f"special kind {kind!r} is excluded from support")
        counts[matches[0]] += value
    vocab_size = (
        max((int(value) for value in tokenizer.get_vocab().values()), default=-1) + 1
    )
    if smoothing > 0:
        active_ids = tuple(
            token_id
            for token_id in range(vocab_size)
            if token_id in allowed and token_id not in excluded_ids
        )
    else:
        active_ids = tuple(
            sorted(token_id for token_id, count in counts.items() if count > 0)
        )
    if not active_ids:
        raise MarginalError("canonical content produced no active tokens")
    count_array = np.zeros(vocab_size, dtype=np.float64)
    for token_id in active_ids:
        count_array[token_id] = counts[token_id] + smoothing
    probabilities = count_array / count_array.sum()
    marginal = TokenMarginal(probabilities, count_array, active_ids)
    marginal.validate()
    return marginal
