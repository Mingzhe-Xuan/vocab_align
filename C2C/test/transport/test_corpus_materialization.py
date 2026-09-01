import json
import subprocess
import sys
from pathlib import Path

import pytest

from rosetta.transport.corpus_materialization import (
    MaterializationError,
    materialize_corpus,
)


REVISION = "d" * 40


def _record(index):
    return {
        "id": None,
        "conversations": [
            {"from": "human", "value": f"question {index}"},
            {"from": "gpt", "value": f"answer {index}"},
        ],
    }


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_materialization_is_atomic_and_records_selection_provenance(tmp_path):
    records = [_record(0), _record(1), _record(1), _record(2), _record(3), _record(4)]
    records_output = tmp_path / "selected.jsonl"
    manifest_output = tmp_path / "manifest.json"
    manifest = materialize_corpus(
        lambda: iter(records),
        records_output,
        manifest_output,
        dataset="fixture/openhermes",
        revision=REVISION,
        sample_count=4,
        dev_fraction=0.25,
    )
    assert len(records_output.read_text(encoding="utf-8").splitlines()) == 4
    materialized = [
        json.loads(line)
        for line in records_output.read_text(encoding="utf-8").splitlines()
    ]
    assert materialized == records[:4]
    assert manifest["selection"] == {
        "algorithm": "pinned-source-prefix-v1",
        "source_start_index": 0,
        "requested_count": 4,
        "selected_source_rows": 4,
        "unique_conversations": 3,
        "adapter_filtering": "not-applied",
        "split_seed": 42,
    }
    assert len(manifest["transport_train"]) == 2
    assert len(manifest["transport_dev"]) == 1
    assert json.loads(manifest_output.read_text(encoding="utf-8")) == manifest

    failed_records = tmp_path / "failed.jsonl"
    failed_manifest = tmp_path / "failed-manifest.json"
    with pytest.raises(MaterializationError, match="expected 8"):
        materialize_corpus(
            lambda: iter(records),
            failed_records,
            failed_manifest,
            dataset="fixture/openhermes",
            revision=REVISION,
            sample_count=8,
        )
    assert not failed_records.exists()
    assert not failed_manifest.exists()
    assert not (tmp_path / "failed.jsonl.partial").exists()


def test_materialization_rejects_unpinned_or_test_inputs(tmp_path):
    factory = lambda: iter([_record(0)])
    with pytest.raises(MaterializationError, match="revision"):
        materialize_corpus(
            factory,
            tmp_path / "records",
            tmp_path / "manifest",
            dataset="fixture",
            revision="main",
            sample_count=1,
        )
    with pytest.raises(MaterializationError, match="non-test"):
        materialize_corpus(
            factory,
            tmp_path / "records",
            tmp_path / "manifest",
            dataset="fixture",
            revision=REVISION,
            raw_split="benchmark_test",
            sample_count=1,
        )


def test_materialization_cli_uses_offline_jsonl_without_datasets_import(tmp_path):
    input_path = tmp_path / "input.jsonl"
    records_output = tmp_path / "selected.jsonl"
    manifest_output = tmp_path / "manifest.json"
    _write_jsonl(input_path, [_record(index) for index in range(5)])
    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "script.dataset.materialize_transport_corpus",
            "--dataset",
            "fixture/openhermes",
            "--dataset-revision",
            REVISION,
            "--input-jsonl",
            str(input_path),
            "--records-output",
            str(records_output),
            "--manifest-output",
            str(manifest_output),
            "--sample-count",
            "3",
            "--dev-fraction",
            "0.34",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
    assert manifest["selection"]["selected_source_rows"] == 3
    assert manifest["dataset_revision"] == REVISION
