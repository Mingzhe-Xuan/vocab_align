import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from rosetta.transport.artifact import load_transport_artifact, save_transport_artifact
from rosetta.transport.audit import audit_transport_artifact, transport_to_dense
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


def test_target_rescue_makes_split_target_marginal_feasible(TinyTokenizer):
    text = "ab"
    source = TinyTokenizer({"ab": 0}, {text: [(0, 0, 2)]})
    target = TinyTokenizer(
        {"ab-unused": 0, "a": 1, "b": 2},
        {text: [(1, 0, 1), (2, 1, 2)]},
    )
    target._by_id[0] = "ab"
    result = build_vocab_transport(
        source,
        target,
        [text],
        epsilon=0.5,
        tolerance=1e-9,
        max_iter=10_000,
        smoothing=0.0,
        seed=42,
        code_version="test",
    )
    assert result.artifact.source_token_ids.tolist() == [0]
    assert result.artifact.target_token_ids.tolist() == [1, 2]
    np.testing.assert_allclose(
        transport_to_dense(result.artifact) @ result.artifact.source_marginal,
        result.artifact.target_marginal,
        atol=1e-9,
    )


def test_positive_smoothing_covers_full_source_and_ordinary_target(TinyTokenizer):
    text = "a"
    control = "<control>"
    source = TinyTokenizer(
        {"a": 0, control: 1},
        {text: [(0, 0, 1)], control: [(1, 0, len(control))]},
        specials=(control,),
    )
    target = TinyTokenizer(
        {"a": 0, "<": 1, "control": 2, ">": 3, "<bos>": 4},
        {
            text: [(0, 0, 1)],
            control: [(1, 0, 1), (2, 1, 8), (3, 8, 9)],
        },
        specials=("<bos>",),
        bos_token_id=4,
    )
    result = build_vocab_transport(
        source,
        target,
        [text],
        epsilon=0.5,
        smoothing=0.1,
        ann_fallback=lambda source_id, raw: [
            (target_id, 1e-6) for target_id in range(4)
        ],
        data_config={"mode": "manifest-bound-canonical-conversations"},
        seed=42,
        code_version="test",
    )
    assert result.artifact.source_token_ids.tolist() == [0, 1]
    assert result.artifact.target_token_ids.tolist() == [0, 1, 2, 3]
    assert (
        result.artifact.metadata["build_config"]["support_policy"]["target"]
        == "ordinary-only"
    )
    assert result.artifact.metadata["build_config"]["data"]["mode"] == (
        "manifest-bound-canonical-conversations"
    )
    report = audit_transport_artifact(result.artifact)
    assert report["valid"]
    assert report["candidate_source_counts"]["special_literal"] == 3
