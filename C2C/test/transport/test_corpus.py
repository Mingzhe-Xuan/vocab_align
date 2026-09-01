import json
import subprocess
import sys
from pathlib import Path

import pytest

from rosetta.transport.corpus import (
    CorpusError,
    build_corpus_manifest,
    canonical_messages,
    canonical_sample_id,
    load_manifest_texts,
)
from rosetta.transport.manifest import save_manifest


REVISION = "c" * 40


def _record(index):
    return {
        "id": None,
        "conversations": [
            {"from": "human", "value": f"question {index}", "weight": None},
            {"from": "gpt", "value": f"answer {index}", "weight": None},
        ],
        "source": "fixture",
    }


def _write_records(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_canonical_messages_roles_ids_and_schema_validation():
    record = {
        "conversations": [
            {"from": "system", "value": "rules"},
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi"},
        ]
    }
    messages = canonical_messages(record)
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert canonical_sample_id(messages) == canonical_sample_id(
        canonical_messages({**record, "unrelated": 1})
    )
    with pytest.raises(CorpusError, match="unsupported"):
        canonical_messages({"conversations": [{"from": "tool", "value": "x"}]})
    with pytest.raises(CorpusError, match="non-empty"):
        canonical_messages({"conversations": [{"from": "human", "value": " "}]})


def test_manifest_deduplicates_content_and_binds_raw_hash(tmp_path):
    records_path = tmp_path / "records.jsonl"
    records = [_record(index) for index in range(4)]
    _write_records(records_path, records + [records[1]])
    manifest = build_corpus_manifest(
        records_path,
        dataset="fixture/openhermes",
        revision=REVISION,
        dev_fraction=0.25,
    )
    assert manifest["raw_record_count"] == 5
    assert manifest["unique_record_count"] == 4
    assert manifest["duplicate_content_records"] == 1
    assert len(manifest["transport_train"]) == 3
    assert len(manifest["transport_dev"]) == 1

    reordered_path = tmp_path / "reordered.jsonl"
    _write_records(reordered_path, list(reversed(records)))
    reordered = build_corpus_manifest(
        reordered_path,
        dataset="fixture/openhermes",
        revision=REVISION,
        dev_fraction=0.25,
    )
    assert reordered["transport_train"] == manifest["transport_train"]
    assert reordered["transport_dev"] == manifest["transport_dev"]
    assert reordered["raw_input_sha256"] != manifest["raw_input_sha256"]

    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)
    texts, provenance = load_manifest_texts(
        records_path, manifest_path, build_split="transport_train"
    )
    assert len(texts) == 6
    assert provenance["selected_samples"] == 3
    assert provenance["canonical_messages"] == 6
    assert provenance["dataset_revision"] == REVISION

    records_path.write_text(
        records_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(CorpusError, match="SHA-256"):
        load_manifest_texts(records_path, manifest_path, build_split="transport_train")


def test_manifest_loader_rejects_split_overlap(tmp_path):
    records_path = tmp_path / "records.jsonl"
    _write_records(records_path, [_record(index) for index in range(3)])
    manifest = build_corpus_manifest(
        records_path,
        dataset="fixture/openhermes",
        revision=REVISION,
        dev_fraction=0.34,
    )
    manifest["transport_dev"] = [manifest["transport_train"][0]]
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)
    with pytest.raises(CorpusError, match="overlap"):
        load_manifest_texts(records_path, manifest_path, build_split="transport_train")


def test_manifest_loader_rejects_sample_omitted_from_both_splits(tmp_path):
    records_path = tmp_path / "records.jsonl"
    _write_records(records_path, [_record(index) for index in range(3)])
    manifest = build_corpus_manifest(
        records_path,
        dataset="fixture/openhermes",
        revision=REVISION,
        dev_fraction=0.34,
    )
    manifest["transport_train"] = manifest["transport_train"][:-1]
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)
    with pytest.raises(CorpusError, match="does not reproduce"):
        load_manifest_texts(records_path, manifest_path, build_split="transport_train")


def test_content_manifest_cli_records_complete_provenance(tmp_path):
    records_path = tmp_path / "records.jsonl"
    manifest_path = tmp_path / "manifest.json"
    _write_records(records_path, [_record(index) for index in range(5)])
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "script.dataset.build_transport_manifest",
            "--input",
            str(records_path),
            "--output",
            str(manifest_path),
            "--identity-mode",
            "canonical-content",
            "--dataset",
            "fixture/openhermes",
            "--dataset-revision",
            REVISION,
            "--dev-fraction",
            "0.2",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["identity_scheme"] == "canonical-conversation-sha256-v1"
    assert manifest["dataset_revision"] == REVISION
    assert len(manifest["raw_input_sha256"]) == 64


@pytest.mark.parametrize("split", ["benchmark_test", "test"])
def test_corpus_rejects_test_splits(tmp_path, split):
    records_path = tmp_path / "records.jsonl"
    _write_records(records_path, [_record(0)])
    with pytest.raises(CorpusError, match="non-test"):
        build_corpus_manifest(
            records_path,
            dataset="fixture",
            revision=REVISION,
            raw_split=split,
        )
