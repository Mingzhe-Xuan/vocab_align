"""Sparse candidate edges with explicit alignment evidence and precedence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .token_metadata import (
    encode_with_byte_spans,
    ordinary_bytes_index,
    special_id_to_kind,
    special_id_to_token,
    token_raw_bytes,
)


class CandidateGraphError(ValueError):
    """Raised when a safe candidate support cannot be constructed."""


class EdgeSource(str, Enum):
    SPECIAL = "special"
    EXACT_BYTE = "exact_byte"
    BYTE_SPAN = "byte_span"
    ANN = "ann"


@dataclass(frozen=True)
class CandidateEdge:
    source_id: int
    target_id: int
    source: EdgeSource
    evidence: float


@dataclass(frozen=True)
class CandidateGraph:
    source_vocab_size: int
    target_vocab_size: int
    edges: Tuple[CandidateEdge, ...]

    def source_ids(self) -> Set[int]:
        return {edge.source_id for edge in self.edges}

    def target_ids(self) -> Set[int]:
        return {edge.target_id for edge in self.edges}

    def validate_support(
        self, required_source_ids: Iterable[int], required_target_ids: Iterable[int]
    ) -> None:
        missing_source = sorted(set(required_source_ids).difference(self.source_ids()))
        missing_target = sorted(set(required_target_ids).difference(self.target_ids()))
        if missing_source:
            raise CandidateGraphError(
                f"positive-mass source tokens lack candidate edges: {missing_source}"
            )
        if missing_target:
            raise CandidateGraphError(
                f"positive-mass target tokens lack candidate edges: {missing_target}"
            )


AnnFallback = Callable[[int, bytes], Sequence[Tuple[int, float]]]


def accumulate_byte_span_counts(
    source_tokenizer: Any, target_tokenizer: Any, texts: Iterable[str]
) -> Dict[int, Counter[int]]:
    counts: Dict[int, Counter[int]] = defaultdict(Counter)
    for text in texts:
        source_spans = encode_with_byte_spans(source_tokenizer, text)
        target_spans = encode_with_byte_spans(target_tokenizer, text)
        target_cursor = 0
        for source_id, source_start, source_end in source_spans:
            while (
                target_cursor < len(target_spans)
                and target_spans[target_cursor][2] <= source_start
            ):
                target_cursor += 1
            cursor = target_cursor
            while cursor < len(target_spans) and target_spans[cursor][1] < source_end:
                target_id, target_start, target_end = target_spans[cursor]
                overlap = min(source_end, target_end) - max(source_start, target_start)
                if overlap > 0:
                    counts[source_id][target_id] += overlap
                cursor += 1
    return dict(counts)


def _special_target(
    source_id: int,
    source_tokenizer: Any,
    target_tokenizer: Any,
) -> int:
    source_kinds = special_id_to_kind(source_tokenizer)
    target_kinds = special_id_to_kind(target_tokenizer)
    kind = source_kinds[source_id]
    if kind == "special":
        source_token = special_id_to_token(source_tokenizer)[source_id]
        matches = [
            target_id
            for target_id, token in special_id_to_token(target_tokenizer).items()
            if token == source_token
        ]
    else:
        matches = [target_id for target_id, target_kind in target_kinds.items() if target_kind == kind]
    if len(matches) != 1:
        raise CandidateGraphError(
            f"required special source token {source_id} has no unambiguous {kind!r} target"
        )
    return matches[0]


def build_candidate_graph(
    source_tokenizer: Any,
    target_tokenizer: Any,
    texts: Iterable[str],
    *,
    required_source_ids: Iterable[int],
    required_target_ids: Iterable[int] = (),
    ann_fallback: AnnFallback | None = None,
) -> CandidateGraph:
    texts = [text for text in texts if text]
    required_source = sorted(set(int(value) for value in required_source_ids))
    required_target = sorted(set(int(value) for value in required_target_ids))
    source_vocab = {int(value) for value in source_tokenizer.get_vocab().values()}
    target_vocab = {int(value) for value in target_tokenizer.get_vocab().values()}
    if not set(required_source).issubset(source_vocab):
        raise CandidateGraphError("required source ID is outside tokenizer vocabulary")
    if not set(required_target).issubset(target_vocab):
        raise CandidateGraphError("required target ID is outside tokenizer vocabulary")

    source_special = special_id_to_kind(source_tokenizer)
    target_special = set(special_id_to_kind(target_tokenizer))
    exact_target = ordinary_bytes_index(target_tokenizer)
    spans = accumulate_byte_span_counts(source_tokenizer, target_tokenizer, texts)
    edges: List[CandidateEdge] = []

    for source_id in required_source:
        if source_id in source_special:
            target_id = _special_target(source_id, source_tokenizer, target_tokenizer)
            edges.append(CandidateEdge(source_id, target_id, EdgeSource.SPECIAL, 1.0))
            continue
        raw_bytes = token_raw_bytes(source_tokenizer, source_id)
        exact_ids = exact_target.get(raw_bytes, [])
        if exact_ids:
            edges.extend(
                CandidateEdge(source_id, target_id, EdgeSource.EXACT_BYTE, 1.0)
                for target_id in sorted(exact_ids)
            )
            continue
        span_candidates = spans.get(source_id, Counter())
        if span_candidates:
            edges.extend(
                CandidateEdge(
                    source_id, target_id, EdgeSource.BYTE_SPAN, float(evidence)
                )
                for target_id, evidence in sorted(span_candidates.items())
                if target_id not in target_special
            )
            if edges and edges[-1].source_id == source_id:
                continue
        if ann_fallback is None:
            raise CandidateGraphError(
                f"source token {source_id} has no exact/span edge or ANN fallback"
            )
        ann_edges = []
        for target_id, score in ann_fallback(source_id, raw_bytes):
            target_id = int(target_id)
            score = float(score)
            if target_id not in target_vocab or target_id in target_special:
                raise CandidateGraphError("ANN fallback returned an unsafe target token")
            if not isfinite(score):
                raise CandidateGraphError("ANN fallback evidence must be finite")
            ann_edges.append(CandidateEdge(source_id, target_id, EdgeSource.ANN, score))
        if not ann_edges:
            raise CandidateGraphError(f"ANN fallback returned no edge for source {source_id}")
        edges.extend(sorted(ann_edges, key=lambda edge: edge.target_id))

    graph = CandidateGraph(
        source_vocab_size=max(source_vocab, default=-1) + 1,
        target_vocab_size=max(target_vocab, default=-1) + 1,
        edges=tuple(sorted(edges, key=lambda edge: (edge.source_id, edge.target_id))),
    )
    graph.validate_support(required_source, required_target)
    return graph
