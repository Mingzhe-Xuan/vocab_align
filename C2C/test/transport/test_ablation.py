import json
import subprocess
import sys

import pytest
import yaml

from rosetta.transport.ablation import (
    AblationError,
    AblationPlan,
    paired_evaluation_summary,
)


def _plan_payload():
    return {
        "schema_version": 1,
        "dev_split": "transport_dev",
        "test_split": "benchmark:test",
        "dimensions": {
            "tau": [0.7, 1.0],
            "causal_shift": [True, False],
            "approximation": [{"mode": "exact"}, {"mode": "hard"}],
        },
        "frozen": {
            "tau": 1.0,
            "causal_shift": True,
            "approximation": {"mode": "exact"},
        },
    }


def test_dev_expansion_is_deterministic_and_test_is_frozen():
    plan = AblationPlan.from_dict(_plan_payload())
    dev = plan.expand("dev")
    assert len(dev) == 8
    assert len({run.run_id for run in dev}) == 8
    assert [run.to_dict() for run in dev] == [
        run.to_dict() for run in plan.expand("dev")
    ]

    test = plan.expand("test")
    assert len(test) == 1
    assert test[0].split == "benchmark:test"
    assert test[0].parameters == _plan_payload()["frozen"]


def test_run_ids_do_not_depend_on_yaml_key_order():
    payload = _plan_payload()
    reordered = {
        **payload,
        "dimensions": dict(reversed(list(payload["dimensions"].items()))),
        "frozen": dict(reversed(list(payload["frozen"].items()))),
    }
    first = AblationPlan.from_dict(payload).expand("dev")
    second = AblationPlan.from_dict(reordered).expand("dev")
    assert [run.run_id for run in first] == [run.run_id for run in second]


def test_invalid_frozen_and_nonfinite_values_are_rejected():
    payload = _plan_payload()
    payload["frozen"] = {**payload["frozen"], "tau": 2.0}
    with pytest.raises(AblationError, match="outside"):
        AblationPlan.from_dict(payload)

    payload = _plan_payload()
    payload["dimensions"] = {**payload["dimensions"], "tau": [float("nan")]}
    with pytest.raises(AblationError, match="finite"):
        AblationPlan.from_dict(payload)


def _record(sample_id, correct, status="success"):
    return {"sample_id": sample_id, "status": status, "is_correct": correct}


def test_paired_summary_reports_missing_samples_instead_of_hiding_them():
    summary = paired_evaluation_summary(
        [_record("a", True), _record("b", False), _record("ignored", False, "failed")],
        [_record("a", False), _record("c", True)],
    )
    assert summary["complete_pairing"] is False
    assert summary["paired_sample_ids"] == ["a"]
    assert summary["missing_candidate_ids"] == ["b"]
    assert summary["missing_reference_ids"] == ["c"]
    assert summary["reference_accuracy"] == 1.0
    assert summary["candidate_accuracy"] == 0.0
    assert summary["accuracy_delta"] == -1.0
    assert summary["reference_wins"] == 1


def test_paired_summary_rejects_duplicate_success():
    with pytest.raises(AblationError, match="duplicate"):
        paired_evaluation_summary(
            [_record("a", True), _record("a", False)], [_record("a", True)]
        )


def test_ablation_cli_writes_atomic_dev_and_frozen_test_plans(tmp_path):
    config = tmp_path / "plan.yaml"
    config.write_text(yaml.safe_dump(_plan_payload()), encoding="utf-8")
    for phase, expected in (("dev", 8), ("test", 1)):
        output = tmp_path / f"{phase}.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "script.transport.run_transport_ablation",
                "--config",
                str(config),
                "--phase",
                phase,
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(output.read_text(encoding="utf-8"))
        assert payload["run_count"] == expected
        assert not output.with_name(output.name + ".partial").exists()


def test_repository_ablation_recipe_covers_required_dimensions():
    payload = yaml.safe_load(
        open("recipe/eval_recipe/stt_ablation.yaml", encoding="utf-8")
    )
    plan = AblationPlan.from_dict(payload)
    assert plan.dev_split == "transport_dev"
    assert "test" in plan.test_split
    assert set(plan.dimensions) == {
        "transport_source",
        "fitting",
        "epsilon",
        "tau",
        "causal_shift",
        "approximation",
    }
    modes = {choice["mode"] for choice in plan.dimensions["approximation"]}
    assert modes == {"exact", "hard", "top_m", "orf"}
    assert len(plan.expand("dev")) == 648
    assert len(plan.expand("test")) == 1
