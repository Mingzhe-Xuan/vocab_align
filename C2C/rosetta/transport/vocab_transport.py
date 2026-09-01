"""Small, auditable vocabulary alignment based on bytes and text spans.

This is deliberately a local column-normalized baseline.  It does not use
ANN fallback or optimal transport, and it only creates columns for source
tokens observed in the supplied texts (plus observed special tokens).
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .artifact import TransportArtifact
from .candidate_graph import AnnFallback, CandidateGraph, EdgeSource, build_candidate_graph
from .marginals import TokenMarginal, estimate_token_marginal
from .sinkhorn import (
    ConvergenceReport,
    SparseCoupling,
    candidate_edge_costs,
    dense_sinkhorn,
    sparse_conditional_from_coupling,
    sparse_log_sinkhorn,
)

from .token_metadata import (
    encode_with_byte_spans,
    special_id_to_token,
    token_raw_bytes,
    tokenizer_fingerprint,
)


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


@dataclass(frozen=True)
class VocabTransportBuildResult:
    artifact: TransportArtifact
    graph: CandidateGraph
    coupling: SparseCoupling
    convergence: ConvergenceReport
    dense_oracle_max_error: float | None


def _compact_artifact(
    graph: CandidateGraph,
    coupling: SparseCoupling,
    source_marginal: TokenMarginal,
    target_marginal: TokenMarginal,
    metadata: Mapping[str, Any],
) -> TransportArtifact:
    source_ids = np.asarray(source_marginal.active_ids, dtype=np.int64)
    target_ids = np.asarray(target_marginal.active_ids, dtype=np.int64)
    source_positions = {token_id: index for index, token_id in enumerate(source_ids)}
    target_positions = {token_id: index for index, token_id in enumerate(target_ids)}
    conditional = sparse_conditional_from_coupling(
        coupling, source_marginal.probabilities
    )
    rows = np.asarray(
        [target_positions[int(value)] for value in conditional.row_indices],
        dtype=np.int64,
    )
    columns = np.asarray(
        [source_positions[int(value)] for value in conditional.column_indices],
        dtype=np.int64,
    )
    order = np.lexsort((rows, columns))
    rows, columns, data = rows[order], columns[order], conditional.data[order]
    indptr = np.concatenate(
        ([0], np.cumsum(np.bincount(columns, minlength=len(source_ids))))
    ).astype(np.int64)

    active_edges = tuple(
        edge
        for edge in graph.edges
        if edge.source_id in source_positions and edge.target_id in target_positions
    )
    candidate_rows = np.asarray(
        [target_positions[edge.target_id] for edge in active_edges], dtype=np.int64
    )
    candidate_columns = np.asarray(
        [source_positions[edge.source_id] for edge in active_edges], dtype=np.int64
    )
    artifact = TransportArtifact(
        indptr=indptr,
        indices=rows,
        data=data,
        shape=(len(target_ids), len(source_ids)),
        source_marginal=source_marginal.probabilities[source_ids],
        target_marginal=target_marginal.probabilities[target_ids],
        metadata=dict(metadata),
        source_token_ids=source_ids,
        target_token_ids=target_ids,
        candidate_rows=candidate_rows,
        candidate_columns=candidate_columns,
        candidate_evidence=np.asarray(
            [edge.evidence for edge in active_edges], dtype=np.float64
        ),
        candidate_sources=np.asarray(
            [edge.source.value for edge in active_edges], dtype="U16"
        ),
    )
    artifact.validate()
    return artifact


def build_vocab_transport(
    source_tokenizer: Any,
    target_tokenizer: Any,
    texts: Iterable[str],
    *,
    epsilon: float,
    tolerance: float = 1e-9,
    max_iter: int = 10_000,
    smoothing: float = 0.0,
    ann_fallback: AnnFallback | None = None,
    seed: int,
    code_version: str,
    dense_oracle_limit: int = 10_000,
    ann_config: Mapping[str, Any] | None = None,
) -> VocabTransportBuildResult:
    """Build an auditable sparse OT transport over active tokenizer support."""
    texts = [text for text in texts if text]
    if not texts:
        raise ValueError("at least one non-empty canonical text is required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not code_version.strip():
        raise ValueError("code_version is required")
    source_fingerprint = tokenizer_fingerprint(source_tokenizer)
    target_fingerprint = tokenizer_fingerprint(target_tokenizer)
    source_marginal = estimate_token_marginal(
        source_tokenizer, texts, smoothing=smoothing
    )
    target_marginal = estimate_token_marginal(
        target_tokenizer, texts, smoothing=smoothing
    )
    graph = build_candidate_graph(
        source_tokenizer,
        target_tokenizer,
        texts,
        required_source_ids=source_marginal.active_ids,
        required_target_ids=target_marginal.active_ids,
        ann_fallback=ann_fallback,
    )
    coupling, convergence = sparse_log_sinkhorn(
        graph,
        source_marginal.probabilities,
        target_marginal.probabilities,
        epsilon=epsilon,
        tolerance=tolerance,
        max_iter=max_iter,
    )
    active_edges = tuple(
        edge
        for edge in graph.edges
        if source_marginal.probabilities[edge.source_id] > 0
        and target_marginal.probabilities[edge.target_id] > 0
    )
    dense_error = None
    active_size = len(source_marginal.active_ids) * len(target_marginal.active_ids)
    if active_size <= dense_oracle_limit:
        source_positions = {
            token_id: index for index, token_id in enumerate(source_marginal.active_ids)
        }
        target_positions = {
            token_id: index for index, token_id in enumerate(target_marginal.active_ids)
        }
        dense_cost = np.full(
            (len(target_positions), len(source_positions)), np.inf, dtype=np.float64
        )
        for edge, cost in zip(active_edges, candidate_edge_costs(active_edges)):
            dense_cost[target_positions[edge.target_id], source_positions[edge.source_id]] = cost
        dense_coupling, _ = dense_sinkhorn(
            dense_cost,
            source_marginal.probabilities[list(source_marginal.active_ids)],
            target_marginal.probabilities[list(target_marginal.active_ids)],
            epsilon=epsilon,
            tolerance=tolerance,
            max_iter=max_iter,
        )
        sparse_dense = np.zeros_like(dense_coupling)
        for row, column, value in zip(
            coupling.row_indices, coupling.column_indices, coupling.data
        ):
            sparse_dense[target_positions[int(row)], source_positions[int(column)]] = value
        dense_error = float(np.max(np.abs(dense_coupling - sparse_dense)))

    fingerprint_payload = json.dumps(
        {
            "source_fingerprint": source_fingerprint,
            "target_fingerprint": target_fingerprint,
            "texts": texts,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata = {
        "schema_version": 1,
        "source_fingerprint": source_fingerprint,
        "target_fingerprint": target_fingerprint,
        "input_fingerprint": hashlib.sha256(fingerprint_payload).hexdigest(),
        "build_config": {
            "epsilon": epsilon,
            "tolerance": tolerance,
            "max_iter": max_iter,
            "smoothing": smoothing,
            "ann": dict(ann_config or {"enabled": ann_fallback is not None}),
        },
        "seed": seed,
        "code_version": code_version,
        "convergence": convergence.to_dict(),
        "dense_oracle_max_error": dense_error,
        "coordinate_system": "active-support-target-by-source",
        "special_mappings": [
            {"source_id": edge.source_id, "target_id": edge.target_id}
            for edge in active_edges
            if edge.source == EdgeSource.SPECIAL
        ],
    }
    artifact = _compact_artifact(
        graph, coupling, source_marginal, target_marginal, metadata
    )
    return VocabTransportBuildResult(
        artifact, graph, coupling, convergence, dense_error
    )


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

    source_special = special_id_to_token(source_tokenizer)
    target_special_by_token = {
        token: token_id for token_id, token in special_id_to_token(target_tokenizer).items()
    }
    target_special_ids = set(special_id_to_token(target_tokenizer))
    target_bytes: DefaultDict[bytes, List[int]] = defaultdict(list)
    for target_id in range(len(target_tokenizer)):
        if target_id not in target_special_ids:
            target_bytes[token_raw_bytes(target_tokenizer, target_id)].append(target_id)

    source_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    span_counts: DefaultDict[int, Counter[int]] = defaultdict(Counter)

    for text in texts:
        source_spans = encode_with_byte_spans(source_tokenizer, text)
        target_spans = encode_with_byte_spans(target_tokenizer, text)
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
        source_fingerprint=tokenizer_fingerprint(source_tokenizer),
        target_fingerprint=tokenizer_fingerprint(target_tokenizer),
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
