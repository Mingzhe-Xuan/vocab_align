import json
import subprocess
import sys

import yaml


def test_summarize_transport_cli_and_pinned_recipe(tmp_path):
    records = tmp_path / "records.jsonl"
    output = tmp_path / "summary.json"
    record = {
        "schema_version": 1,
        "sample_id": "abstract_algebra:0",
        "subject": "abstract_algebra",
        "question_index": 0,
        "method": "training_free_transport",
        "status": "success",
        "is_correct": True,
        "metrics": {"total_seconds": 1.0},
    }
    records.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "script.transport.summarize_transport",
            "--records",
            str(records),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["accuracy"] == 1.0

    recipe = yaml.safe_load(
        open("recipe/eval_recipe/stt_mmlu_redux.yaml", encoding="utf-8")
    )
    assert recipe["model"]["model_name"] == "training_free_transport"
    assert recipe["model"]["runtime_profile"] == "blackwell-cu128"
    assert recipe["model"]["generation_config"]["do_sample"] is False
    assert recipe["eval"]["dataset"] == "mmlu-redux"
    assert recipe["eval"]["subjects"] == ["abstract_algebra"]
    assert recipe["eval"]["limit"] == 5


def test_unified_evaluator_help_does_not_require_optional_math_packages():
    result = subprocess.run(
        [sys.executable, "-m", "script.evaluation.unified_evaluator", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_summarizer_reports_incomplete_pairing(tmp_path):
    reference = tmp_path / "reference.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "summary.json"
    base = {
        "schema_version": 1,
        "subject": "math",
        "question_index": 0,
        "method": "exact",
        "status": "success",
        "is_correct": True,
        "metrics": {},
    }
    reference.write_text(
        json.dumps({**base, "sample_id": "a"})
        + "\n"
        + json.dumps({**base, "sample_id": "missing"})
        + "\n",
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps({**base, "sample_id": "a", "method": "hard"}) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "script.transport.summarize_transport",
            "--records",
            str(candidate),
            "--reference-records",
            str(reference),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    paired = json.loads(output.read_text(encoding="utf-8"))["paired_comparison"]
    assert paired["complete_pairing"] is False
    assert paired["missing_candidate_ids"] == ["missing"]
