import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from rosetta.transport.artifact import load_transport_artifact, save_transport_artifact
from rosetta.transport.audit import audit_transport_artifact
from rosetta.transport.vocab_transport import build_vocab_transport


def test_toy_facade_dense_sparse_artifact_and_audit(TinyTokenizer, tmp_path):
    text = "abcd"
    source = TinyTokenizer(
        {"ab": 0, "cd": 1},
        {text: [(0, 0, 2), (1, 2, 4)]},
    )
    source.name_or_path = "toy-source"
    target = TinyTokenizer(
        {"a": 0, "b": 1, "c": 2, "d": 3},
        {text: [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 4)]},
    )
    target.name_or_path = "toy-target"
    result = build_vocab_transport(
        source,
        target,
        [text],
        epsilon=0.5,
        seed=42,
        code_version="toy-commit",
    )
    assert result.dense_oracle_max_error is not None
    assert result.dense_oracle_max_error < 1e-10
    artifact_path = tmp_path / "toy.npz"
    save_transport_artifact(result.artifact, artifact_path)
    loaded = load_transport_artifact(
        artifact_path,
        source_fingerprint=result.artifact.metadata["source_fingerprint"],
        target_fingerprint=result.artifact.metadata["target_fingerprint"],
    )
    np.testing.assert_array_equal(loaded.source_token_ids, [0, 1])
    np.testing.assert_array_equal(loaded.target_token_ids, [0, 1, 2, 3])
    assert len(loaded.candidate_rows) == 4
    report = audit_transport_artifact(loaded)
    assert report["valid"]
    assert report["candidate_source_counts"] == {"byte_span": 4}
    assert report["max_column_sum_error"] < 1e-10
    assert report["transported_marginal_l1"] < 1e-10

    json_output = tmp_path / "audit.json"
    markdown_output = tmp_path / "audit.md"
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "script.transport.audit_vocab_transport",
            "--artifact",
            str(artifact_path),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(json_output.read_text(encoding="utf-8"))["valid"]
    assert "Vocabulary transport audit" in markdown_output.read_text(encoding="utf-8")


def test_facade_compacts_zero_mass_tokens_and_preserves_original_ids(TinyTokenizer):
    text = "xy"
    source = TinyTokenizer(
        {"unused": 0, "xy": 1},
        {text: [(1, 0, 2)]},
    )
    target = TinyTokenizer(
        {"unused-a": 0, "unused-b": 1, "x": 2, "y": 3},
        {text: [(2, 0, 1), (3, 1, 2)]},
    )
    result = build_vocab_transport(
        source,
        target,
        [text],
        epsilon=0.5,
        seed=42,
        code_version="toy-commit",
    )
    assert result.artifact.shape == (2, 1)
    np.testing.assert_array_equal(result.artifact.source_token_ids, [1])
    np.testing.assert_array_equal(result.artifact.target_token_ids, [2, 3])
