import json
import subprocess
import sys
from pathlib import Path

import pytest

from script.transport.build_vocab_transport import _load_ann_candidates


def test_toy_build_cli_is_atomic_audited_and_resumable(tmp_path):
    root = Path(__file__).resolve().parents[2]
    artifact = tmp_path / "toy.npz"
    audit_json = tmp_path / "audit.json"
    audit_markdown = tmp_path / "audit.md"
    command = [
        sys.executable,
        "-m",
        "script.transport.build_vocab_transport",
        "--toy",
        "--artifact",
        str(artifact),
        "--audit-json",
        str(audit_json),
        "--audit-markdown",
        str(audit_markdown),
        "--code-version",
        "toy-test",
    ]
    first = subprocess.run(
        command, cwd=root, capture_output=True, text=True, check=False
    )
    assert first.returncode == 0, first.stderr
    assert artifact.exists()
    assert not artifact.with_suffix(".npz.partial.npz").exists()
    checkpoint = json.loads(
        artifact.with_suffix(".npz.checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "complete"
    assert json.loads(audit_json.read_text(encoding="utf-8"))["valid"]
    second = subprocess.run(
        command + ["--resume"], cwd=root, capture_output=True, text=True, check=False
    )
    assert second.returncode == 0, second.stderr
    resumed = json.loads(
        artifact.with_suffix(".npz.checkpoint.json").read_text(encoding="utf-8")
    )
    assert resumed["resume"] == "loaded-valid-artifact"


def test_ann_candidate_loader_accepts_structured_and_legacy_json(tmp_path):
    structured = {
        "schema_version": 1,
        "input_fingerprint": "input",
        "source_fingerprint": "source",
        "target_fingerprint": "target",
        "build_config": {"method": "test"},
        "seed": 42,
        "code_version": "version",
        "coverage": {"source_tokens_with_candidates": 1},
        "candidates": {"0": [[1, 0.5]]},
    }
    structured_path = tmp_path / "structured.json"
    structured_path.write_text(json.dumps(structured), encoding="utf-8")
    candidates, config = _load_ann_candidates(structured_path)
    assert candidates == structured["candidates"]
    assert config["provenance"]["input_fingerprint"] == "input"
    assert len(config["sha256"]) == 64

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text('{"0": [[1, 0.5]]}', encoding="utf-8")
    candidates, config = _load_ann_candidates(legacy_path)
    assert candidates == {"0": [[1, 0.5]]}
    assert config["provenance"] == {"schema_version": 0, "kind": "legacy-mapping"}


def test_ann_candidate_loader_rejects_incomplete_provenance(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"schema_version": 1, "candidates": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="provenance missing"):
        _load_ann_candidates(path)
