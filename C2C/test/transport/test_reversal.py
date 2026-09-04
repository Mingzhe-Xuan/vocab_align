import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosetta.transport.artifact import (
    artifact_from_dense,
    load_transport_artifact,
    save_transport_artifact,
)
from rosetta.transport.audit import audit_transport_artifact, transport_to_dense
from rosetta.transport.reversal import REVERSAL_METHOD, reverse_transport_artifact


def _forward_artifact():
    transport = np.array([[0.5, 0.2], [0.3, 0.5], [0.2, 0.3]], dtype=np.float64)
    source_marginal = np.array([0.4, 0.6], dtype=np.float64)
    recorded_target = np.array([0.3205, 0.4195, 0.26], dtype=np.float64)
    metadata = {
        "schema_version": 1,
        "source_fingerprint": "source-fingerprint",
        "target_fingerprint": "target-fingerprint",
        "input_fingerprint": "parent-input",
        "build_config": {
            "epsilon": 0.5,
            "tolerance": 0.002,
            "support_policy": {
                "source": "full-vocabulary-under-positive-smoothing",
                "target": "ordinary-only",
                "source_special_fallback": "target-tokenized-literal-bytes",
            },
            "ann": {"enabled": True, "kind": "fixture"},
        },
        "seed": 42,
        "code_version": "parent-version",
        "convergence": {"converged": True, "row_residual": 0.001},
        "special_mappings": [{"source_id": 10, "target_id": 20}],
    }
    artifact = artifact_from_dense(
        transport, source_marginal, recorded_target, metadata
    )
    rows, columns = np.nonzero(transport)
    labels = np.full(len(rows), "exact_byte", dtype="U16")
    labels[0] = "special"
    return replace(
        artifact,
        source_token_ids=np.array([10, 12]),
        target_token_ids=np.array([20, 21, 23]),
        candidate_rows=rows.astype(np.int64),
        candidate_columns=columns.astype(np.int64),
        candidate_evidence=np.linspace(1.0, 2.0, len(rows)),
        candidate_sources=labels,
    )


def test_bayes_reversal_preserves_joint_mass_and_swaps_direction():
    forward = _forward_artifact()
    reverse = reverse_transport_artifact(
        forward, code_version="reverse-version", parent_sha256="a" * 64
    )

    assert reverse.shape == (2, 3)
    np.testing.assert_array_equal(reverse.source_token_ids, [20, 21, 23])
    np.testing.assert_array_equal(reverse.target_token_ids, [10, 12])
    assert reverse.metadata["source_fingerprint"] == "target-fingerprint"
    assert reverse.metadata["target_fingerprint"] == "source-fingerprint"
    assert reverse.metadata["derivation"]["fingerprint_validation"] == "not-performed"
    assert reverse.metadata["build_config"]["derivation"] == REVERSAL_METHOD
    assert reverse.metadata["special_mappings"] == [{"source_id": 20, "target_id": 10}]

    forward_dense = transport_to_dense(forward)
    reverse_dense = transport_to_dense(reverse)
    forward_joint = forward_dense * forward.source_marginal[None, :]
    reverse_joint = reverse_dense * reverse.source_marginal[None, :]
    np.testing.assert_allclose(reverse_joint, forward_joint.T, atol=1e-15)
    np.testing.assert_allclose(reverse.source_marginal, forward_joint.sum(axis=1))
    np.testing.assert_allclose(reverse_dense.sum(axis=0), 1.0, atol=1e-15)
    assert reverse.metadata["derivation"][
        "parent_recorded_target_vs_realized_l1"
    ] == pytest.approx(0.001)
    assert audit_transport_artifact(reverse)["valid"] is True


def test_double_reversal_recovers_transport_marginals_support_and_candidates():
    forward = _forward_artifact()
    reverse = reverse_transport_artifact(forward, code_version="first")
    restored = reverse_transport_artifact(reverse, code_version="second")

    np.testing.assert_allclose(
        transport_to_dense(restored), transport_to_dense(forward), atol=1e-15
    )
    np.testing.assert_allclose(restored.source_marginal, forward.source_marginal)
    np.testing.assert_allclose(
        restored.target_marginal,
        transport_to_dense(forward) @ forward.source_marginal,
    )
    np.testing.assert_array_equal(restored.source_token_ids, forward.source_token_ids)
    np.testing.assert_array_equal(restored.target_token_ids, forward.target_token_ids)
    np.testing.assert_array_equal(restored.candidate_rows, forward.candidate_rows)
    np.testing.assert_array_equal(restored.candidate_columns, forward.candidate_columns)


def test_reverse_cli_is_atomic_audited_and_does_not_check_live_fingerprints(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "forward.npz"
    output = tmp_path / "reverse.npz"
    audit_json = tmp_path / "reverse.json"
    audit_markdown = tmp_path / "reverse.md"
    save_transport_artifact(_forward_artifact(), source)
    command = [
        sys.executable,
        "-m",
        "script.transport.reverse_vocab_transport",
        "--input",
        str(source),
        "--artifact",
        str(output),
        "--audit-json",
        str(audit_json),
        "--audit-markdown",
        str(audit_markdown),
        "--code-version",
        "cli-test",
    ]
    process = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )

    assert process.returncode == 0, process.stderr
    assert output.exists()
    assert not output.with_suffix(".npz.partial.npz").exists()
    report = json.loads(audit_json.read_text(encoding="utf-8"))
    assert report["valid"] is True
    loaded = load_transport_artifact(output)
    assert loaded.metadata["code_version"] == "cli-test"
    assert loaded.metadata["derivation"]["parent_sha256"]
    assert audit_markdown.read_text(encoding="utf-8").startswith(
        "# Vocabulary transport audit"
    )

    repeated = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )
    assert repeated.returncode != 0
    assert "refusing to overwrite" in repeated.stderr
