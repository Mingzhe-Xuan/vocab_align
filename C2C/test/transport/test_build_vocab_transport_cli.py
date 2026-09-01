import json
import subprocess
import sys
from pathlib import Path


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
    first = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
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
