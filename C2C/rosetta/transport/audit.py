"""Independent invariant and quality audit for transport artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .artifact import ArtifactError, TransportArtifact
from .candidate_graph import EdgeSource


def transport_to_dense(artifact: TransportArtifact) -> np.ndarray:
    dense = np.zeros(artifact.shape, dtype=np.float64)
    for column in range(artifact.shape[1]):
        start, end = artifact.indptr[column : column + 2]
        dense[artifact.indices[start:end], column] = artifact.data[start:end]
    return dense


def audit_transport_artifact(artifact: TransportArtifact) -> Dict[str, Any]:
    artifact.validate()
    target_size, source_size = artifact.shape
    column_sums = np.zeros(source_size, dtype=np.float64)
    transported = np.zeros(target_size, dtype=np.float64)
    column_entropy = np.zeros(source_size, dtype=np.float64)

    candidate_cost_keys = np.asarray([], dtype=np.int64)
    candidate_cost_values = np.asarray([], dtype=np.float64)
    if len(artifact.candidate_rows):
        labels, label_counts = np.unique(artifact.candidate_sources, return_counts=True)
        try:
            for label in labels:
                EdgeSource(str(label))
        except ValueError as exc:
            raise ArtifactError(
                "candidate graph contains an unknown source label"
            ) from exc
        source_counts = {
            str(label): int(count) for label, count in zip(labels, label_counts)
        }
        candidate_columns = artifact.candidate_columns.astype(np.int64, copy=False)
        candidate_rows = artifact.candidate_rows.astype(np.int64, copy=False)
        evidence = artifact.candidate_evidence.astype(np.float64, copy=False)
        evidence_totals = np.bincount(
            candidate_columns, weights=evidence, minlength=source_size
        )
        evidence_counts = np.bincount(candidate_columns, minlength=source_size)
        costs = -np.log(
            (evidence + 1e-12)
            / (
                evidence_totals[candidate_columns]
                + 1e-12 * evidence_counts[candidate_columns]
            )
        )
        candidate_keys = candidate_columns * target_size + candidate_rows
        order = np.argsort(candidate_keys)
        candidate_cost_keys = candidate_keys[order]
        candidate_cost_values = costs[order]
        if np.any(np.diff(candidate_cost_keys) == 0):
            raise ArtifactError("candidate graph contains duplicate row/column edges")
    else:
        source_counts = {}

    transport_cost = 0.0 if len(candidate_cost_keys) else None
    entropy_term = 0.0
    for column in range(source_size):
        start, end = artifact.indptr[column : column + 2]
        rows = artifact.indices[start:end]
        values = artifact.data[start:end].astype(np.float64, copy=False)
        column_sums[column] = values.sum()
        coupling_values = values * artifact.source_marginal[column]
        np.add.at(transported, rows, coupling_values)
        positive = values > 0
        if np.any(positive):
            positive_values = values[positive]
            column_entropy[column] = -float(
                np.dot(positive_values, np.log(positive_values))
            )
            positive_coupling = coupling_values[coupling_values > 0]
            if len(positive_coupling):
                entropy_term += float(
                    np.dot(positive_coupling, np.log(positive_coupling) - 1.0)
                )
        if transport_cost is not None:
            keys = column * target_size + rows
            positions = np.searchsorted(candidate_cost_keys, keys)
            if np.any(positions == len(candidate_cost_keys)) or not np.array_equal(
                candidate_cost_keys[positions], keys
            ):
                raise ArtifactError("transport edge is missing from candidate graph")
            transport_cost += float(
                np.dot(coupling_values, candidate_cost_values[positions])
            )

    row_residual = float(np.abs(transported - artifact.target_marginal).sum())
    column_residual = float(
        np.abs(column_sums * artifact.source_marginal - artifact.source_marginal).sum()
    )
    dangerous_special = []
    special_indices = np.flatnonzero(
        artifact.candidate_sources == EdgeSource.SPECIAL.value
    )
    candidate_special_pairs = {
        (
            int(artifact.target_token_ids[artifact.candidate_rows[index]]),
            int(artifact.source_token_ids[artifact.candidate_columns[index]]),
        )
        for index in special_indices
    }
    for mapping in artifact.metadata.get("special_mappings", []):
        pair = (int(mapping["target_id"]), int(mapping["source_id"]))
        if pair not in candidate_special_pairs:
            dangerous_special.append(mapping)

    regularized_objective = None
    if transport_cost is not None:
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
        "max_column_sum_error": float(np.max(np.abs(column_sums - 1.0))),
        "row_marginal_l1": row_residual,
        "column_marginal_l1": column_residual,
        "transported_marginal_l1": row_residual,
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


def save_audit(
    report: Dict[str, Any], json_path: str | Path, markdown_path: str | Path
) -> None:
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(audit_markdown(report), encoding="utf-8")
