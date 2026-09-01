"""Independent invariant and quality audit for transport artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .artifact import TransportArtifact
from .sinkhorn import candidate_edge_costs
from .candidate_graph import CandidateEdge, EdgeSource


def transport_to_dense(artifact: TransportArtifact) -> np.ndarray:
    dense = np.zeros(artifact.shape, dtype=np.float64)
    for column in range(artifact.shape[1]):
        start, end = artifact.indptr[column : column + 2]
        dense[artifact.indices[start:end], column] = artifact.data[start:end]
    return dense


def audit_transport_artifact(artifact: TransportArtifact) -> Dict[str, Any]:
    artifact.validate()
    transport = transport_to_dense(artifact)
    coupling = transport * artifact.source_marginal[None, :]
    row_residual = float(
        np.abs(coupling.sum(axis=1) - artifact.target_marginal).sum()
    )
    column_residual = float(
        np.abs(coupling.sum(axis=0) - artifact.source_marginal).sum()
    )
    positive = transport > 0
    column_entropy = -np.sum(
        np.where(positive, transport * np.log(np.where(positive, transport, 1.0)), 0.0),
        axis=0,
    )
    source_counts = Counter(str(value) for value in artifact.candidate_sources.tolist())
    dangerous_special = []
    candidate_special_pairs = {
        (
            int(artifact.target_token_ids[row]),
            int(artifact.source_token_ids[column]),
        )
        for row, column, source in zip(
            artifact.candidate_rows,
            artifact.candidate_columns,
            artifact.candidate_sources,
        )
        if str(source) == EdgeSource.SPECIAL.value
    }
    for mapping in artifact.metadata.get("special_mappings", []):
        pair = (int(mapping["target_id"]), int(mapping["source_id"]))
        if pair not in candidate_special_pairs:
            dangerous_special.append(mapping)

    transport_cost = None
    regularized_objective = None
    if len(artifact.candidate_rows):
        edges = tuple(
            CandidateEdge(
                int(column),
                int(row),
                EdgeSource(str(source)),
                float(evidence),
            )
            for row, column, evidence, source in zip(
                artifact.candidate_rows,
                artifact.candidate_columns,
                artifact.candidate_evidence,
                artifact.candidate_sources,
            )
        )
        costs = candidate_edge_costs(edges)
        cost_by_pair = {
            (edge.target_id, edge.source_id): cost for edge, cost in zip(edges, costs)
        }
        transport_cost = 0.0
        for column in range(artifact.shape[1]):
            start, end = artifact.indptr[column : column + 2]
            for row, value in zip(artifact.indices[start:end], coupling[artifact.indices[start:end], column]):
                transport_cost += float(value) * float(cost_by_pair[(int(row), column)])
        entropy_term = float(
            np.sum(np.where(coupling > 0, coupling * (np.log(np.where(coupling > 0, coupling, 1.0)) - 1), 0.0))
        )
        epsilon = float(artifact.metadata["build_config"]["epsilon"])
        regularized_objective = transport_cost + epsilon * entropy_term

    return {
        "schema_version": 1,
        "artifact_schema_version": artifact.metadata["schema_version"],
        "input_fingerprint": artifact.metadata.get("input_fingerprint"),
        "shape": list(artifact.shape),
        "nnz": int(len(artifact.data)),
        "candidate_edge_count": int(len(artifact.candidate_rows)),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "nonnegative": bool(np.all(artifact.data >= 0)),
        "minimum_value": float(artifact.data.min(initial=0.0)),
        "max_column_sum_error": float(np.max(np.abs(transport.sum(axis=0) - 1.0))),
        "row_marginal_l1": row_residual,
        "column_marginal_l1": column_residual,
        "transported_marginal_l1": float(
            np.abs(transport @ artifact.source_marginal - artifact.target_marginal).sum()
        ),
        "column_entropy_quantiles": {
            "q0": float(np.quantile(column_entropy, 0.0)),
            "q50": float(np.quantile(column_entropy, 0.5)),
            "q100": float(np.quantile(column_entropy, 1.0)),
        },
        "transport_cost": transport_cost,
        "regularized_objective": regularized_objective,
        "dangerous_special_mappings": dangerous_special,
        "convergence": artifact.metadata.get("convergence"),
        "dense_oracle_max_error": artifact.metadata.get("dense_oracle_max_error"),
        "valid": not dangerous_special,
    }


def audit_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Vocabulary transport audit",
        "",
        f"- Valid: `{str(report['valid']).lower()}`",
        f"- Shape: `{report['shape'][0]} x {report['shape'][1]}`",
        f"- Nonzeros: `{report['nnz']}`",
        f"- Candidate edges: `{report['candidate_edge_count']}`",
        f"- Max column-sum error: `{report['max_column_sum_error']:.6g}`",
        f"- Row marginal L1: `{report['row_marginal_l1']:.6g}`",
        f"- Column marginal L1: `{report['column_marginal_l1']:.6g}`",
        f"- Dense oracle max error: `{report['dense_oracle_max_error']}`",
        "",
        "## Candidate sources",
        "",
    ]
    lines.extend(
        f"- {source}: `{count}`"
        for source, count in report["candidate_source_counts"].items()
    )
    return "\n".join(lines) + "\n"


def save_audit(report: Dict[str, Any], json_path: str | Path, markdown_path: str | Path) -> None:
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(audit_markdown(report), encoding="utf-8")
