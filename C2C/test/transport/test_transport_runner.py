import json
from types import SimpleNamespace

import pytest

from rosetta.transport.evaluation import GenerationResult
from script.evaluation import transport_runner


class _Adapter:
    method = "training_free_transport"

    def generate_one(self, sample):
        return GenerationResult(
            text="Answer: A",
            token_ids=[1],
            metrics={"total_seconds": 1.0},
            diagnostics={"support_mass": 1.0},
        )


def _evaluator(tmp_path):
    calls = []

    def format_example(example, **kwargs):
        calls.append(kwargs)
        return f"Question: {example['question']}"

    evaluator = SimpleNamespace(
        model_config={"model_name": "training_free_transport"},
        output_config={"output_dir": str(tmp_path)},
        eval_config={
            "answer_method": "generate",
            "gpu_ids": [0],
            "sample_interval": 1,
            "limit": 1,
            "subjects": ["abstract_algebra"],
            "use_cot": False,
            "use_template": True,
        },
        dataset_name="mmlu-redux",
        dataset_config={
            "dataset_name": "fixture/mmlu",
            "test_split": "test",
            "subjects": ["abstract_algebra", "anatomy"],
        },
        parse_answer=lambda example: example["answer"],
        extract_predicted_answer=lambda text: text.rsplit(" ", 1)[-1],
        format_example=format_example,
        format_calls=calls,
    )
    return evaluator


def test_subject_samples_preserve_canonical_prompt_metadata(monkeypatch, tmp_path):
    evaluator = _evaluator(tmp_path)
    monkeypatch.setattr(
        transport_runner,
        "load_dataset",
        lambda *args: {
            "test": [
                {"question": "2 + 2?", "answer": "A"},
                {"question": "3 + 3?", "answer": "B"},
            ]
        },
    )

    samples = transport_runner._subject_samples(evaluator, "abstract_algebra")
    assert len(samples) == 1
    assert samples[0].sample_id == "mmlu-redux:abstract_algebra:0"
    assert samples[0].canonical_messages == [
        {"role": "user", "content": "Question: 2 + 2?"}
    ]
    assert samples[0].prompt_metadata["use_template"] is True
    assert evaluator.format_calls == [{"use_cot": False, "use_template": True}]


@pytest.mark.parametrize(
    "dataset_name,subject,expected_args",
    [
        ("mmlu-redux", "abstract_algebra", ("fixture/data", "abstract_algebra")),
        ("gsm8k", "main", ("fixture/data", "main")),
        ("math-500", "all", ("fixture/data",)),
        ("longbench", "qasper", ("fixture/data", "qasper")),
    ],
)
def test_benchmark_loaders_use_explicit_dataset_configs(
    monkeypatch, dataset_name, subject, expected_args
):
    evaluator = SimpleNamespace(
        dataset_name=dataset_name,
        dataset_config={"dataset_name": "fixture/data"},
        eval_config={},
    )
    calls = []
    monkeypatch.setattr(
        transport_runner,
        "load_dataset",
        lambda *args: calls.append(args) or {"test": []},
    )
    transport_runner._load_subject_dataset(evaluator, subject)
    assert calls == [expected_args]


def test_local_benchmark_file_is_hash_checked_and_loaded_by_format(
    monkeypatch, tmp_path
):
    data = tmp_path / "test.jsonl"
    data.write_text('{"answer": "A"}\n', encoding="utf-8")
    from rosetta.transport.corpus import file_sha256

    evaluator = SimpleNamespace(
        dataset_name="math-500",
        dataset_config={"dataset_name": "remote", "test_split": "test"},
        eval_config={
            "data_file": str(data),
            "data_file_sha256": file_sha256(data),
            "data_format": "json",
        },
    )
    calls = []
    monkeypatch.setattr(
        transport_runner,
        "load_dataset",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"test": []},
    )
    transport_runner._load_subject_dataset(evaluator, "all")
    assert calls == [(("json",), {"data_files": {"test": str(data)}})]
    evaluator.eval_config["data_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        transport_runner._load_subject_dataset(evaluator, "all")


def test_longbench_sample_preserves_external_scorer_inputs(monkeypatch, tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator.dataset_name = "longbench"
    evaluator.dataset_config = {
        "dataset_name": "fixture/longbench",
        "test_split": "test",
        "subjects": ["qasper"],
    }
    evaluator._format_longbench_example = lambda example, tokenizer: "Long prompt"
    monkeypatch.setattr(
        transport_runner,
        "load_dataset",
        lambda *args: {
            "test": [
                {
                    "answers": ["reference"],
                    "all_classes": [],
                    "length": 123,
                    "_id": "row-1",
                }
            ]
        },
    )
    sample = transport_runner._subject_samples(
        evaluator, "qasper", source_tokenizer=object()
    )[0]
    assert sample.scoring_mode == "external"
    assert sample.true_answer == "<external-longbench-scorer>"
    assert sample.prompt_metadata["source_prompt_rendered"] is True
    assert sample.prompt_metadata["longbench"] == {
        "answers": ["reference"],
        "all_classes": [],
        "length": 123,
        "id": "row-1",
    }


def test_transport_runner_writes_records_and_summary(monkeypatch, tmp_path):
    evaluator = _evaluator(tmp_path)
    sample = transport_runner.EvaluationSample(
        sample_id="mmlu-redux:abstract_algebra:0",
        subject="abstract_algebra",
        question_index=0,
        canonical_messages=[{"role": "user", "content": "Question"}],
        prompt="Question",
        true_answer="A",
    )
    monkeypatch.setattr(transport_runner.torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(
        transport_runner,
        "create_training_free_transport_adapter",
        lambda _: _Adapter(),
    )
    monkeypatch.setattr(transport_runner, "_subject_samples", lambda *args: [sample])

    transport_runner.run_training_free_transport_evaluation(evaluator)

    records = (tmp_path / "stt.records.jsonl").read_text(encoding="utf-8")
    assert json.loads(records)["status"] == "success"
    summary = json.loads((tmp_path / "stt.summary.json").read_text(encoding="utf-8"))
    assert summary["accuracy"] == 1.0
    assert summary["dataset"] == "mmlu-redux"
