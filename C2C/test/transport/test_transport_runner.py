import json
from types import SimpleNamespace

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
