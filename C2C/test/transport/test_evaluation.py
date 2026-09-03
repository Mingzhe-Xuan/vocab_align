import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from rosetta.transport.evaluation import (
    EvaluationError,
    EvaluationSample,
    GenerationResult,
    evaluate_samples,
    latest_evaluation_records,
    merge_evaluation_records,
    save_evaluation_summary,
    summarize_evaluation_records,
)
from rosetta.transport.metrics import TransportMetrics
from rosetta.transport.soft_transport import SoftTransportStats
from rosetta.transport.wrapper import TransportGenerationOutput
from script.evaluation.transport_adapter import (
    TrainingFreeTransportEvaluationAdapter,
    create_training_free_transport_adapter,
)


def _sample(sample_id="algebra:0", prompt="rendered prompt"):
    return EvaluationSample(
        sample_id=sample_id,
        subject="abstract_algebra",
        question_index=int(sample_id.rsplit(":", 1)[-1]),
        canonical_messages=[{"role": "user", "content": "2 + 2?"}],
        prompt=prompt,
        true_answer="A",
        prompt_metadata={"template": "target-chat-v1", "few_shot": 0},
    )


@dataclass
class StubAdapter:
    method: str
    fail_ids: set[str]

    def __post_init__(self):
        self.calls = []

    def generate_one(self, sample):
        self.calls.append(sample.sample_id)
        if sample.sample_id in self.fail_ids:
            raise RuntimeError("synthetic generation failure")
        return GenerationResult(
            text="Answer: A",
            token_ids=[10, 11],
            metrics={
                "source_seconds": 0.1,
                "transport_seconds": 0.2,
                "receiver_seconds": 0.3,
                "source_tokens": 4,
                "virtual_tokens": 4,
            },
            diagnostics={"support_mass": 1.0},
        )


def _parse_answer(text):
    return text.rsplit(" ", 1)[-1] if text else None


@pytest.mark.parametrize(
    "method", ["receiver", "source", "t2t", "c2c", "training_free_transport"]
)
def test_methods_share_versioned_record_schema(tmp_path, method):
    sample = _sample()
    records = tmp_path / f"{method}.jsonl"
    result = evaluate_samples(
        [sample],
        StubAdapter(method, set()),
        _parse_answer,
        records,
        tmp_path / "bad.jsonl",
    )[0]

    assert result["schema_version"] == 1
    assert result["sample_id"] == sample.sample_id
    assert result["canonical_messages"] == list(sample.canonical_messages)
    assert result["prompt_metadata"] == sample.prompt_metadata
    assert result["method"] == method
    assert result["metrics"]["transport_seconds"] == pytest.approx(0.2)
    assert result["status"] == "success"


def test_failure_is_logged_then_retried_without_counting_as_wrong(tmp_path):
    records = tmp_path / "records.jsonl"
    bad = tmp_path / "bad.jsonl"
    sample0, sample1 = _sample("algebra:0"), _sample("algebra:1")
    first = StubAdapter("training_free_transport", {sample0.sample_id})

    attempts = evaluate_samples([sample0, sample1], first, _parse_answer, records, bad)
    assert [record["status"] for record in attempts] == ["failed", "success"]
    assert json.loads(bad.read_text(encoding="utf-8"))["sample_id"] == sample0.sample_id
    first_summary = summarize_evaluation_records(attempts)
    assert first_summary["successful_samples"] == 1
    assert first_summary["failed_samples"] == 1
    assert first_summary["accuracy"] == 1.0

    second = StubAdapter("training_free_transport", set())
    resumed = evaluate_samples([sample0, sample1], second, _parse_answer, records, bad)
    assert second.calls == [sample0.sample_id]
    assert all(record["status"] == "success" for record in resumed)
    assert set(latest_evaluation_records(records)) == {
        sample0.sample_id,
        sample1.sample_id,
    }


def test_resume_rejects_changed_prompt_or_method(tmp_path):
    records = tmp_path / "records.jsonl"
    sample = _sample()
    evaluate_samples(
        [sample],
        StubAdapter("receiver", set()),
        _parse_answer,
        records,
        tmp_path / "bad.jsonl",
    )

    with pytest.raises(EvaluationError, match="prompt fingerprint changed"):
        evaluate_samples(
            [_sample(prompt="changed")],
            StubAdapter("receiver", set()),
            _parse_answer,
            records,
            tmp_path / "bad.jsonl",
        )
    with pytest.raises(EvaluationError, match="method changed"):
        evaluate_samples(
            [sample],
            StubAdapter("source", set()),
            _parse_answer,
            records,
            tmp_path / "bad.jsonl",
        )


def test_duplicate_success_is_rejected(tmp_path):
    records = tmp_path / "records.jsonl"
    sample = _sample()
    record = evaluate_samples(
        [sample],
        StubAdapter("receiver", set()),
        _parse_answer,
        records,
        tmp_path / "bad.jsonl",
    )[0]
    with records.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    with pytest.raises(EvaluationError, match="duplicate successful"):
        latest_evaluation_records(records)


def test_merge_is_deterministic_and_rejects_cross_rank_duplicates(tmp_path):
    rank0, rank1 = tmp_path / "rank0.jsonl", tmp_path / "rank1.jsonl"
    bad = tmp_path / "bad.jsonl"
    evaluate_samples(
        [_sample("algebra:1")],
        StubAdapter("receiver", set()),
        _parse_answer,
        rank0,
        bad,
    )
    evaluate_samples(
        [_sample("algebra:0")],
        StubAdapter("receiver", set()),
        _parse_answer,
        rank1,
        bad,
    )

    merged = merge_evaluation_records([rank0, rank1], tmp_path / "merged.jsonl")
    assert [record["sample_id"] for record in merged] == ["algebra:0", "algebra:1"]
    with pytest.raises(EvaluationError, match="duplicate sample across ranks"):
        merge_evaluation_records([rank0, rank0], tmp_path / "invalid.jsonl")


def test_summary_aggregates_metrics_failures_and_saves_atomically(tmp_path):
    success = {
        "status": "success",
        "sample_id": "a",
        "subject": "math",
        "method": "stt",
        "is_correct": True,
        "metrics": {"total_seconds": 2.0, "tokens_per_second": 3.0},
    }
    failure = {
        "status": "failed",
        "sample_id": "b",
        "error": {"type": "RuntimeError", "message": "failed"},
    }

    summary = summarize_evaluation_records([success, failure])
    assert summary["accuracy"] == 1.0
    assert summary["metric_means"]["total_seconds"] == 2.0
    assert summary["failure_reasons"] == {"RuntimeError": 1}
    output = tmp_path / "summary.json"
    save_evaluation_summary(summary, output)
    assert json.loads(output.read_text(encoding="utf-8")) == summary
    assert not (tmp_path / "summary.json.partial").exists()


def test_external_scoring_success_is_not_counted_as_incorrect(tmp_path):
    sample = replace(
        _sample(),
        true_answer="<external-scorer>",
        scoring_mode="external",
    )
    records = evaluate_samples(
        [sample],
        StubAdapter("training_free_transport", set()),
        lambda _: "must-not-be-used",
        tmp_path / "records.jsonl",
        tmp_path / "bad.jsonl",
    )
    assert records[0]["is_correct"] is None
    assert records[0]["scoring_status"] == "external_required"
    summary = summarize_evaluation_records(records)
    assert summary["successful_samples"] == 1
    assert summary["scored_samples"] == 0
    assert summary["unscored_samples"] == 1
    assert summary["accuracy"] is None
    assert summary["subjects"]["abstract_algebra"]["accuracy"] is None


def test_sample_rejects_non_integer_question_index():
    sample = _sample()
    invalid = EvaluationSample(
        sample.sample_id,
        sample.subject,
        "0",  # type: ignore[arg-type]
        sample.canonical_messages,
        sample.prompt,
        sample.true_answer,
    )
    with pytest.raises(EvaluationError, match="nonnegative integer"):
        invalid.validate()


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "2 + 2?"}]
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "rendered source prompt"

    def __call__(self, text, **kwargs):
        assert text == "rendered source prompt"
        assert kwargs == {"return_tensors": "pt"}
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, token_ids, *, skip_special_tokens):
        assert token_ids == [3]
        assert skip_special_tokens is True
        return "Answer: A"


class _Wrapper:
    def __init__(self):
        self.source_model = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros((4, 2)))
        )

    def generate(self, source_ids, **kwargs):
        assert source_ids.tolist() == [[1, 2]]
        assert kwargs["source_attention_mask"].tolist() == [[1, 1]]
        assert kwargs["return_transport_output"] is True
        assert kwargs["max_new_tokens"] == 1
        return TransportGenerationOutput(
            sequences=torch.tensor([[3]]),
            virtual_prompt_shape=(1, 2, 4),
            stats=SoftTransportStats(
                retained_mass=torch.ones((1, 2)),
                dropped_top_m_mass=torch.zeros((1, 2)),
                active_support_mass=torch.ones((1, 2)),
                top_m=None,
            ),
            metrics=TransportMetrics(0.1, 0.2, 0.3, 0.4, 1.0, 2, 2, 1, None),
        )


def test_training_free_adapter_emits_transport_metrics_and_diagnostics():
    adapter = TrainingFreeTransportEvaluationAdapter(
        _Wrapper(),
        _Tokenizer(),
        _Tokenizer(),
        {"max_new_tokens": 1},
        {"code_version": "fixture"},
    )
    result = adapter.generate_one(_sample())
    assert result.text == "Answer: A"
    assert result.metrics["transport_seconds"] == pytest.approx(0.2)
    assert result.diagnostics["virtual_prompt_shape"] == [1, 2, 4]
    assert result.diagnostics["active_support_mass_mean"] == 1.0
    assert result.diagnostics["source_prompt_rendered"] is False
    assert result.diagnostics["provenance"] == {"code_version": "fixture"}


def test_training_free_adapter_encodes_pre_rendered_source_prompt_once():
    class PreRenderedTokenizer(_Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            raise AssertionError(
                "pre-rendered source prompt must not be rendered again"
            )

    sample = replace(
        _sample(prompt="rendered source prompt"),
        prompt_metadata={"source_prompt_rendered": True},
    )
    result = TrainingFreeTransportEvaluationAdapter(
        _Wrapper(),
        PreRenderedTokenizer(),
        _Tokenizer(),
        {"max_new_tokens": 1},
    ).generate_one(sample)
    assert result.diagnostics["source_prompt_rendered"] is True
    assert result.diagnostics["source_rendered_prompt"] == "rendered source prompt"


def test_training_free_adapter_rejects_non_boolean_rendered_prompt_marker():
    sample = replace(_sample(), prompt_metadata={"source_prompt_rendered": "true"})
    adapter = TrainingFreeTransportEvaluationAdapter(
        _Wrapper(), _Tokenizer(), _Tokenizer(), {"max_new_tokens": 1}
    )
    with pytest.raises(ValueError, match="must be a boolean"):
        adapter.generate_one(sample)


def test_training_free_adapter_marks_orf_transport_stats_unavailable():
    wrapper = _Wrapper()
    wrapper.approximation_mode = "orf"
    original_generate = wrapper.generate

    def generate_without_stats(*args, **kwargs):
        return replace(original_generate(*args, **kwargs), stats=None)

    wrapper.generate = generate_without_stats
    result = TrainingFreeTransportEvaluationAdapter(
        wrapper, _Tokenizer(), _Tokenizer(), {"max_new_tokens": 1}
    ).generate_one(_sample())
    assert result.diagnostics["approximation_mode"] == "orf"
    assert result.diagnostics["transport_stats_available"] is False
    assert result.diagnostics["retained_mass_mean"] is None
    assert result.diagnostics["active_support_mass_mean"] is None


def test_training_free_config_creates_adapter(monkeypatch):
    loaded = {}

    monkeypatch.setattr(
        "script.evaluation.transport_adapter.validate_runtime_requirements",
        lambda *args, **kwargs: loaded.setdefault("runtime", (args, kwargs)),
    )
    monkeypatch.setattr(
        "script.evaluation.transport_adapter.runtime_metadata",
        lambda profile: {"profile": profile},
    )
    monkeypatch.setattr(
        "script.evaluation.transport_adapter.file_sha256", lambda _: "artifact-sha"
    )
    monkeypatch.setattr(
        "script.evaluation.transport_adapter._git_version", lambda: "code-sha"
    )
    wrapper = _Wrapper()
    wrapper.artifact = SimpleNamespace(
        shape=(2, 3),
        data=SimpleNamespace(size=4),
        metadata={"input_fingerprint": "fingerprint"},
    )

    def fake_load_with_artifact(config, artifact, **kwargs):
        loaded["config"] = config
        loaded["artifact"] = artifact
        loaded["device_maps"] = kwargs
        return wrapper, _Tokenizer(), _Tokenizer()

    monkeypatch.setattr(
        "script.evaluation.transport_adapter._load_runtime", fake_load_with_artifact
    )
    monkeypatch.setattr(
        "script.evaluation.transport_adapter._configure_approximation",
        lambda candidate, config: loaded.setdefault("approximation", dict(config))
        and candidate,
    )
    adapter = create_training_free_transport_adapter(
        {
            "model_name": "training_free_transport",
            "transport_config": (
                "recipe/transport_recipe/"
                "qwen3_8b_to_mistral_nemo_instruct_2407_smoke.yaml"
            ),
            "artifact": "local/fake.npz",
            "generation_config": {"max_new_tokens": 1},
            "source_device_map": "cpu",
            "target_device_map": "auto",
            "approximation": {"mode": "top_m", "source_top_m": 256},
        }
    )
    assert adapter.method == "training_free_transport"
    assert adapter.generation["max_new_tokens"] == 1
    assert loaded["artifact"].as_posix() == "local/fake.npz"
    assert loaded["runtime"][1]["allow_existing_output"] is True
    assert adapter.provenance["code_version"] == "code-sha"
    assert adapter.provenance["runtime"]["profile"] == "project-cu124"
    assert adapter.provenance["artifact_sha256"] == "artifact-sha"
    assert adapter.provenance["artifact_shape"] == [2, 3]
    assert loaded["approximation"] == {"mode": "top_m", "source_top_m": 256}
    assert adapter.provenance["approximation"] == {
        "mode": "top_m",
        "source_top_m": 256,
    }
    assert loaded["device_maps"] == {
        "source_device_map": "cpu",
        "target_device_map": "auto",
    }
    assert adapter.provenance["source_device_map_override"] == "cpu"
