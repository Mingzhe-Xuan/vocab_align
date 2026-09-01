import argparse
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from rosetta.transport.corpus import build_corpus_manifest
from rosetta.transport.manifest import save_manifest
from script.transport.build_vocab_transport import _load_ann_candidates, _real_inputs


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


def test_real_inputs_loads_only_manifest_bound_canonical_split(tmp_path, monkeypatch):
    records_path = tmp_path / "records.jsonl"
    records = [
        {
            "conversations": [
                {"from": "human", "value": f"question {index}"},
                {"from": "gpt", "value": f"answer {index}"},
            ]
        }
        for index in range(4)
    ]
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = build_corpus_manifest(
        records_path,
        dataset="fixture/openhermes",
        revision="d" * 40,
        dev_fraction=0.25,
    )
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return {"name": name, **kwargs}

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    args = argparse.Namespace(
        source="source/model",
        target="target/model",
        source_revision="a" * 40,
        target_revision="b" * 40,
        texts_jsonl=None,
        records_jsonl=records_path,
        manifest_json=manifest_path,
        build_split="transport_train",
        ann_candidates_json=None,
    )
    source, target, texts, fallback, ann, data = _real_inputs(args)
    assert source["revision"] == "a" * 40
    assert target["revision"] == "b" * 40
    assert len(texts) == 6
    assert fallback is None
    assert ann == {"enabled": False}
    assert data["build_split"] == "transport_train"
    assert data["selected_samples"] == 3

    args.texts_jsonl = records_path
    with pytest.raises(ValueError, match="cannot mix"):
        _real_inputs(args)
