import json
import subprocess
import sys
from pathlib import Path

import pytest

from rosetta.transport.manifest import (
    ManifestError,
    build_transport_manifest,
    serialize_manifest,
)


def test_manifest_is_byte_stable_and_order_independent():
    ids = [f"sample-{index:03d}" for index in range(100)]
    first = build_transport_manifest(ids, seed=42, dev_fraction=0.1)
    second = build_transport_manifest(reversed(ids), seed=42, dev_fraction=0.1)
    assert serialize_manifest(first) == serialize_manifest(second)
    assert len(first["transport_train"]) == 90
    assert len(first["transport_dev"]) == 10
    assert not set(first["transport_train"]).intersection(first["transport_dev"])


def test_manifest_rejects_duplicates():
    with pytest.raises(ManifestError, match="duplicate"):
        build_transport_manifest(["same", "same"])


def test_manifest_cli_tiny_fixture(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "manifest.json"
    input_path.write_text(
        "".join(json.dumps({"id": f"id-{i}"}) + "\n" for i in range(10)),
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "script.dataset.build_transport_manifest",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--dev-fraction",
            "0.2",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sample_count"] == 10
    assert len(payload["transport_dev"]) == 2
