import json
from dataclasses import replace

import numpy as np
import pytest

from rosetta.transport.artifact import (
    ArtifactError,
    artifact_from_dense,
    load_transport_artifact,
    save_transport_artifact,
)


def _metadata():
    return {
        "schema_version": 1,
        "source_fingerprint": "source-sha256",
        "target_fingerprint": "target-sha256",
        "build_config": {"epsilon": 0.5},
        "seed": 42,
        "code_version": "test-commit",
    }


def test_artifact_round_trip_preserves_sparse_arrays_and_dtype(tmp_path):
    transport = np.array([[1.0, 0.25], [0.0, 0.75]], dtype=np.float32)
    source = np.array([0.4, 0.6], dtype=np.float64)
    target = transport @ source
    artifact = artifact_from_dense(transport, source, target, _metadata())
    path = tmp_path / "transport.npz"
    save_transport_artifact(artifact, path)
    loaded = load_transport_artifact(
        path,
        source_fingerprint="source-sha256",
        target_fingerprint="target-sha256",
    )

    assert loaded.shape == artifact.shape
    assert loaded.data.dtype == np.float32
    np.testing.assert_array_equal(loaded.indptr, artifact.indptr)
    np.testing.assert_array_equal(loaded.indices, artifact.indices)
    np.testing.assert_array_equal(loaded.data, artifact.data)
    assert loaded.metadata == artifact.metadata


def test_artifact_rejects_schema_fingerprint_and_nonfinite_values(tmp_path):
    transport = np.eye(2)
    marginal = np.array([0.5, 0.5])
    metadata = _metadata()
    metadata["schema_version"] = 99
    with pytest.raises(ArtifactError, match="schema"):
        artifact_from_dense(transport, marginal, marginal, metadata)

    artifact = artifact_from_dense(transport, marginal, marginal, _metadata())
    path = tmp_path / "transport.npz"
    save_transport_artifact(artifact, path)
    with pytest.raises(ArtifactError, match="fingerprint"):
        load_transport_artifact(path, source_fingerprint="different")

    artifact.data[0] = np.nan
    with pytest.raises(ArtifactError, match="finite"):
        artifact.validate()


def test_artifact_rejects_missing_arrays_and_unnormalized_marginal(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, data=np.array([1.0]))
    with pytest.raises(ArtifactError, match="arrays missing"):
        load_transport_artifact(path)

    with pytest.raises(ArtifactError, match="sum to one"):
        artifact_from_dense(
            np.eye(2),
            np.array([0.25, 0.25]),
            np.array([0.25, 0.25]),
            _metadata(),
        )


def test_artifact_rejects_corrupt_candidate_arrays():
    marginal = np.array([0.5, 0.5])
    artifact = artifact_from_dense(np.eye(2), marginal, marginal, _metadata())
    corrupt = replace(
        artifact,
        candidate_rows=np.array([0]),
        candidate_columns=np.array([], dtype=np.int64),
    )
    with pytest.raises(ArtifactError, match="candidate graph arrays"):
        corrupt.validate()


def test_legacy_schema_one_artifact_loads_with_identity_token_ids(tmp_path):
    path = tmp_path / "legacy.npz"
    metadata = json.dumps(_metadata(), sort_keys=True, separators=(",", ":"))
    np.savez_compressed(
        path,
        indptr=np.array([0, 1, 2], dtype=np.int64),
        indices=np.array([0, 1], dtype=np.int64),
        data=np.array([1.0, 1.0]),
        shape=np.array([2, 2], dtype=np.int64),
        source_marginal=np.array([0.5, 0.5]),
        target_marginal=np.array([0.5, 0.5]),
        metadata=np.asarray(metadata),
    )
    loaded = load_transport_artifact(path)
    np.testing.assert_array_equal(loaded.source_token_ids, [0, 1])
    np.testing.assert_array_equal(loaded.target_token_ids, [0, 1])
    assert len(loaded.candidate_rows) == 0
