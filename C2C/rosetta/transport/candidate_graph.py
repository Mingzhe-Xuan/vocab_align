"""Sparse candidate edges with explicit alignment evidence and precedence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

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
    SPECIAL_LITERAL = "special_literal"
    EXACT_BYTE = "exact_byte"
    BYTE_SPAN = "byte_span"
    ANN = "ann"
    FEASIBILITY = "feasibility"


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


def augment_candidate_graph_for_marginals(
    graph: CandidateGraph,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    *,
    evidence: float = 1e-8,
    interior_fraction: float = 0.5,
) -> Tuple[CandidateGraph, int]:
    """Add deterministic low-evidence edges with strictly feasible capacity.

    A small positive mass is first reserved on every existing active edge.  A
    northwest-corner construction then couples the residual marginals.  Adding
    its missing pairs guarantees that the final support admits a coupling that
    is positive on every active edge, which is stronger than graph connectivity.
    """
    source = np.asarray(source_marginal, dtype=np.float64)
    target = np.asarray(target_marginal, dtype=np.float64)
    if source.shape != (graph.source_vocab_size,) or target.shape != (
        graph.target_vocab_size,
    ):
        raise CandidateGraphError(
            "marginal shapes must match candidate graph vocabularies"
        )
    if (
        not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or np.any(source < 0)
        or np.any(target < 0)
    ):
        raise CandidateGraphError("marginals must be finite and nonnegative")
    if not np.isclose(source.sum(), 1.0, rtol=1e-12, atol=1e-12) or not np.isclose(
        target.sum(), 1.0, rtol=1e-12, atol=1e-12
    ):
        raise CandidateGraphError("source and target marginals must each sum to one")
    if not np.isfinite(evidence) or not 0 < evidence < 1e-6:
        raise CandidateGraphError(
            "feasibility evidence must be finite, positive, and below ANN bridge evidence"
        )
    if not np.isfinite(interior_fraction) or not 0 < interior_fraction < 1:
        raise CandidateGraphError("interior fraction must be between zero and one")

    active_source = np.flatnonzero(source > 0)
    active_target = np.flatnonzero(target > 0)
    active_source_set = set(active_source.tolist())
    active_target_set = set(active_target.tolist())
    pairs = set()
    source_degree = np.zeros_like(source, dtype=np.int64)
    target_degree = np.zeros_like(target, dtype=np.int64)
    for edge in graph.edges:
        if not 0 <= edge.source_id < graph.source_vocab_size or not (
            0 <= edge.target_id < graph.target_vocab_size
        ):
            raise CandidateGraphError("candidate edge is outside graph vocabulary")
        pair = (edge.source_id, edge.target_id)
        if pair in pairs:
            raise CandidateGraphError("duplicate candidate edge")
        pairs.add(pair)
        if edge.source_id in active_source_set and edge.target_id in active_target_set:
            source_degree[edge.source_id] += 1
            target_degree[edge.target_id] += 1
    if np.any(source_degree[active_source] == 0) or np.any(
        target_degree[active_target] == 0
    ):
        raise CandidateGraphError("positive-mass token lacks an active candidate edge")

    source_capacity = source[active_source] / source_degree[active_source]
    target_capacity = target[active_target] / target_degree[active_target]
    interior_mass = interior_fraction * float(
        min(source_capacity.min(), target_capacity.min())
    )
    residual_source = source.copy()
    residual_target = target.copy()
    for source_id, target_id in pairs:
        if source_id in active_source_set and target_id in active_target_set:
            residual_source[source_id] -= interior_mass
            residual_target[target_id] -= interior_mass

    additions: List[CandidateEdge] = []
    source_index = target_index = 0
    while source_index < len(active_source) and target_index < len(active_target):
        source_id = int(active_source[source_index])
        target_id = int(active_target[target_index])
        source_mass = residual_source[source_id]
        target_mass = residual_target[target_id]
        if min(source_mass, target_mass) > 0 and (source_id, target_id) not in pairs:
            additions.append(
                CandidateEdge(
                    source_id,
                    target_id,
                    EdgeSource.FEASIBILITY,
                    float(evidence),
                )
            )
            pairs.add((source_id, target_id))
        if source_mass < target_mass:
            residual_target[target_id] -= source_mass
            residual_source[source_id] = 0.0
            source_index += 1
        elif target_mass < source_mass:
            residual_source[source_id] -= target_mass
            residual_target[target_id] = 0.0
            target_index += 1
        else:
            residual_source[source_id] = 0.0
            residual_target[target_id] = 0.0
            source_index += 1
            target_index += 1
    if residual_source.sum() > 1e-10 or residual_target.sum() > 1e-10:
        raise CandidateGraphError("failed to construct residual feasible support")
    if not additions:
        return graph, 0
    augmented = CandidateGraph(
        graph.source_vocab_size,
        graph.target_vocab_size,
        tuple(
            sorted(
                (*graph.edges, *additions),
                key=lambda edge: (edge.source_id, edge.target_id),
            )
        ),
    )
    return augmented, len(additions)


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
        matches = [
            target_id
            for target_id, target_kind in target_kinds.items()
            if target_kind == kind
        ]
    if len(matches) != 1:
        raise CandidateGraphError(
            f"required special source token {source_id} has no unambiguous {kind!r} target"
        )
    return matches[0]


def _special_literal_candidates(
    source_id: int,
    source_tokenizer: Any,
    target_tokenizer: Any,
) -> Counter[int]:
    source_token = special_id_to_token(source_tokenizer)[source_id]
    target_special = set(special_id_to_kind(target_tokenizer))
    candidates: Counter[int] = Counter()
    for target_id, start, end in encode_with_byte_spans(target_tokenizer, source_token):
        if target_id not in target_special:
            candidates[target_id] += end - start
    return candidates


def build_candidate_graph(
    source_tokenizer: Any,
    target_tokenizer: Any,
    texts: Iterable[str],
    *,
    required_source_ids: Iterable[int],
    required_target_ids: Iterable[int] = (),
    ann_fallback: AnnFallback | None = None,
    special_literal_fallback: bool = False,
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
    exact_source = ordinary_bytes_index(source_tokenizer)
    spans = accumulate_byte_span_counts(source_tokenizer, target_tokenizer, texts)
    edges: List[CandidateEdge] = []
    edge_keys: Set[Tuple[int, int]] = set()

    def add_edge(edge: CandidateEdge) -> None:
        key = (edge.source_id, edge.target_id)
        if key in edge_keys:
            raise CandidateGraphError("duplicate candidate edge")
        edge_keys.add(key)
        edges.append(edge)

    for source_id in required_source:
        if source_id in source_special:
            if special_literal_fallback:
                try:
                    target_id = _special_target(
                        source_id, source_tokenizer, target_tokenizer
                    )
                except CandidateGraphError:
                    pass
                else:
                    add_edge(
                        CandidateEdge(source_id, target_id, EdgeSource.SPECIAL, 1.0)
                    )
                literal = _special_literal_candidates(
                    source_id, source_tokenizer, target_tokenizer
                )
                if not literal:
                    raise CandidateGraphError(
                        f"special source token {source_id} has no ordinary literal target"
                    )
                for target_id, evidence in sorted(literal.items()):
                    if (source_id, target_id) not in edge_keys:
                        add_edge(
                            CandidateEdge(
                                source_id,
                                target_id,
                                EdgeSource.SPECIAL_LITERAL,
                                float(evidence),
                            )
                        )
            else:
                target_id = _special_target(
                    source_id, source_tokenizer, target_tokenizer
                )
                add_edge(CandidateEdge(source_id, target_id, EdgeSource.SPECIAL, 1.0))
            continue
        raw_bytes = token_raw_bytes(source_tokenizer, source_id)
        exact_ids = exact_target.get(raw_bytes, [])
        base_edge_added = False
        if exact_ids:
            for target_id in sorted(exact_ids):
                add_edge(
                    CandidateEdge(source_id, target_id, EdgeSource.EXACT_BYTE, 1.0)
                )
            base_edge_added = True
        else:
            span_candidates = spans.get(source_id, Counter())
            if span_candidates:
                for target_id, evidence in sorted(span_candidates.items()):
                    if target_id in target_special:
                        continue
                    add_edge(
                        CandidateEdge(
                            source_id,
                            target_id,
                            EdgeSource.BYTE_SPAN,
                            float(evidence),
                        )
                    )
                base_edge_added = bool(edges and edges[-1].source_id == source_id)
        ann_edge_added = False
        if ann_fallback is not None:
            ann_targets: Set[int] = set()
            for target_id, score in ann_fallback(source_id, raw_bytes):
                target_id = int(target_id)
                score = float(score)
                if target_id in ann_targets:
                    raise CandidateGraphError("duplicate ANN candidate edge")
                ann_targets.add(target_id)
                if target_id not in target_vocab or target_id in target_special:
                    raise CandidateGraphError(
                        "ANN fallback returned an unsafe target token"
                    )
                if not isfinite(score) or score <= 0:
                    raise CandidateGraphError(
                        "ANN fallback evidence must be finite and positive"
                    )
                if (source_id, target_id) in edge_keys:
                    continue
                add_edge(CandidateEdge(source_id, target_id, EdgeSource.ANN, score))
                ann_edge_added = True
        if not base_edge_added and not ann_edge_added:
            if ann_fallback is not None:
                raise CandidateGraphError(
                    f"ANN fallback returned no usable edge for source {source_id}"
                )
            raise CandidateGraphError(
                f"source token {source_id} has no exact/span edge or ANN fallback"
            )

    missing_targets = sorted(
        set(required_target).difference(edge.target_id for edge in edges)
    )
    source_required = set(required_source)
    target_kinds = special_id_to_kind(target_tokenizer)
    source_tokens = special_id_to_token(source_tokenizer)
    target_tokens = special_id_to_token(target_tokenizer)
    for target_id in missing_targets:
        if target_id in target_kinds:
            kind = target_kinds[target_id]
            special_sources = [
                source_id
                for source_id in required_source
                if source_id in source_special
                and source_special[source_id] == kind
                and (
                    kind != "special"
                    or source_tokens[source_id] == target_tokens[target_id]
                )
            ]
            if len(special_sources) == 1:
                add_edge(
                    CandidateEdge(
                        special_sources[0], target_id, EdgeSource.SPECIAL, 1.0
                    )
                )
                continue
        else:
            raw_bytes = token_raw_bytes(target_tokenizer, target_id)
            exact_sources = [
                source_id
                for source_id in exact_source.get(raw_bytes, [])
                if source_id in source_required and source_id not in source_special
            ]
            if exact_sources:
                for source_id in sorted(exact_sources):
                    add_edge(
                        CandidateEdge(source_id, target_id, EdgeSource.EXACT_BYTE, 1.0)
                    )
                continue
            span_sources = [
                (source_id, counts[target_id])
                for source_id, counts in spans.items()
                if source_id in source_required
                and source_id not in source_special
                and counts[target_id] > 0
            ]
            if span_sources:
                for source_id, evidence in sorted(span_sources):
                    add_edge(
                        CandidateEdge(
                            source_id,
                            target_id,
                            EdgeSource.BYTE_SPAN,
                            float(evidence),
                        )
                    )
                continue
        raise CandidateGraphError(
            f"positive-mass target token {target_id} has no safe rescue edge"
        )

    graph = CandidateGraph(
        source_vocab_size=max(source_vocab, default=-1) + 1,
        target_vocab_size=max(target_vocab, default=-1) + 1,
        edges=tuple(sorted(edges, key=lambda edge: (edge.source_id, edge.target_id))),
    )
    graph.validate_support(required_source, required_target)
    return graph
