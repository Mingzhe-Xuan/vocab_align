import numpy as np

from rosetta.transport.artifact import artifact_from_dense
from rosetta.transport.audit import audit_transport_artifact


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
