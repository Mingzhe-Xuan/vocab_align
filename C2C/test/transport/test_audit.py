from dataclasses import replace

import numpy as np
import pytest

from rosetta.transport.artifact import TransportArtifact, artifact_from_dense
from rosetta.transport.audit import audit_transport_artifact
from rosetta.transport.candidate_graph import EdgeSource


def test_audit_reports_unproven_special_mapping_as_dangerous():
    metadata = {
        "schema_version": 1,
        "source_fingerprint": "source",
        "target_fingerprint": "target",
        "build_config": {"epsilon": 1.0},
        "seed": 42,
        "code_version": "commit",
        "special_mappings": [{"source_id": 0, "target_id": 0}],
    }
    artifact = artifact_from_dense(
        np.eye(2), np.array([0.5, 0.5]), np.array([0.5, 0.5]), metadata
    )
    report = audit_transport_artifact(artifact)
    assert report["valid"] is False
    assert report["dangerous_special_mappings"] == metadata["special_mappings"]


def test_sparse_audit_matches_manual_invariants_entropy_and_objective():
    transport = np.array([[0.8, 0.25], [0.2, 0.75]])
    source = np.array([0.4, 0.6])
    target = transport @ source
    metadata = {
        "schema_version": 1,
        "source_fingerprint": "source",
        "target_fingerprint": "target",
        "build_config": {"epsilon": 0.5, "tolerance": 1e-9},
        "seed": 42,
        "code_version": "commit",
    }
    artifact = artifact_from_dense(transport, source, target, metadata)
    candidate_rows = np.array([1, 0, 1, 0])
    candidate_columns = np.array([1, 0, 0, 1])
    candidate_evidence = np.array([3.0, 4.0, 1.0, 1.0])
    artifact = replace(
        artifact,
        candidate_rows=candidate_rows,
        candidate_columns=candidate_columns,
        candidate_evidence=candidate_evidence,
        candidate_sources=np.full(4, EdgeSource.BYTE_SPAN.value),
    )

    report = audit_transport_artifact(artifact)
    coupling = transport * source[None, :]
    costs = np.array([-np.log(0.75), -np.log(0.8), -np.log(0.2), -np.log(0.25)])
    candidate_coupling = coupling[candidate_rows, candidate_columns]
    expected_cost = float(np.dot(candidate_coupling, costs))
    positive_coupling = coupling[coupling > 0]
    expected_entropy_term = float(
        np.dot(positive_coupling, np.log(positive_coupling) - 1.0)
    )
    column_entropy = -np.sum(transport * np.log(transport), axis=0)

    assert report["row_marginal_l1"] == pytest.approx(0.0)
    assert report["column_marginal_l1"] == pytest.approx(0.0)
    assert report["transported_marginal_l1"] == pytest.approx(0.0)
    assert report["max_column_sum_error"] == pytest.approx(0.0)
    assert report["transport_cost"] == pytest.approx(expected_cost)
    assert report["regularized_objective"] == pytest.approx(
        expected_cost + 0.5 * expected_entropy_term
    )
    assert report["column_entropy_quantiles"]["q0"] == pytest.approx(
        np.quantile(column_entropy, 0.0)
    )
    assert report["column_entropy_quantiles"]["q50"] == pytest.approx(
        np.quantile(column_entropy, 0.5)
    )
    assert report["column_entropy_quantiles"]["q100"] == pytest.approx(
        np.quantile(column_entropy, 1.0)
    )


def test_large_sparse_audit_does_not_call_dense_conversion(monkeypatch):
    size = 10_000
    marginal = np.full(size, 1.0 / size)
    metadata = {
        "schema_version": 1,
        "source_fingerprint": "source",
        "target_fingerprint": "target",
        "build_config": {"epsilon": 0.5, "tolerance": 1e-9},
        "seed": 42,
        "code_version": "commit",
    }
    artifact = TransportArtifact(
        indptr=np.arange(size + 1, dtype=np.int64),
        indices=np.arange(size, dtype=np.int64),
        data=np.ones(size),
        shape=(size, size),
        source_marginal=marginal,
        target_marginal=marginal,
        metadata=metadata,
        source_token_ids=np.arange(size, dtype=np.int64),
        target_token_ids=np.arange(size, dtype=np.int64),
    )

    def fail_dense(_artifact):
        raise AssertionError("full-vocabulary audit must stay sparse")

    monkeypatch.setattr("rosetta.transport.audit.transport_to_dense", fail_dense)
    report = audit_transport_artifact(artifact)
    assert report["shape"] == [size, size]
    assert report["nnz"] == size
    assert report["valid"] is True
