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
