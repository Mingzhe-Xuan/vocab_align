"""Versioned per-sample evaluation records shared by transport methods."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


class EvaluationError(ValueError):
    """Raised when evaluation inputs or persisted records are inconsistent."""


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_finite(value: Any, path: str = "metrics") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise EvaluationError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_finite(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_finite(item, f"{path}[{index}]")
        return
    raise EvaluationError(f"{path} contains unsupported value {type(value).__name__}")


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    subject: str
    question_index: int
    canonical_messages: Sequence[Mapping[str, str]]
    prompt: str
    true_answer: str
    prompt_metadata: Mapping[str, Any] = field(default_factory=dict)
    scoring_mode: str = "exact"

    def validate(self) -> None:
        if not self.sample_id or not self.subject or not self.prompt:
            raise EvaluationError("sample_id, subject, and prompt must be nonempty")
        if (
            isinstance(self.question_index, bool)
            or not isinstance(self.question_index, int)
            or self.question_index < 0
        ):
            raise EvaluationError("question_index must be a nonnegative integer")
        if not isinstance(self.true_answer, str) or not self.true_answer:
            raise EvaluationError("true_answer must be a nonempty string")
        if self.scoring_mode not in {"exact", "external"}:
            raise EvaluationError("scoring_mode must be exact or external")
        if not self.canonical_messages:
            raise EvaluationError("canonical_messages must be nonempty")
        for message in self.canonical_messages:
            if set(message) != {"role", "content"}:
                raise EvaluationError("canonical message requires role and content")
            if not all(isinstance(message[key], str) for key in ("role", "content")):
                raise EvaluationError("canonical message values must be strings")

    @property
    def prompt_fingerprint(self) -> str:
        self.validate()
        payload = {
            "canonical_messages": [
                dict(message) for message in self.canonical_messages
            ],
            "prompt": self.prompt,
            "prompt_metadata": dict(self.prompt_metadata),
        }
        # Preserve stage-3 exact-record hashes; external scoring is a new protocol.
        if self.scoring_mode != "exact":
            payload["scoring_mode"] = self.scoring_mode
        return _canonical_hash(payload)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: Sequence[int]
    metrics: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.text, str):
            raise EvaluationError("generated text must be a string")
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in self.token_ids
        ):
            raise EvaluationError("generated token IDs must be nonnegative integers")
        _validate_finite(self.metrics)
        _validate_finite(self.diagnostics, "diagnostics")


class EvaluationAdapter(Protocol):
    method: str

    def generate_one(self, sample: EvaluationSample) -> GenerationResult: ...


def _base_record(sample: EvaluationSample, method: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sample_id": sample.sample_id,
        "subject": sample.subject,
        "question_index": sample.question_index,
        "method": method,
        "canonical_messages": [dict(message) for message in sample.canonical_messages],
        "prompt": sample.prompt,
        "prompt_fingerprint": sample.prompt_fingerprint,
        "prompt_metadata": dict(sample.prompt_metadata),
        "true_answer": sample.true_answer,
        "scoring_mode": sample.scoring_mode,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise EvaluationError(f"record at {path}:{line_number} must be an object")
        records.append(record)
    return records


def latest_evaluation_records(path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in _read_jsonl(path):
        if record.get("schema_version") != 1:
            raise EvaluationError("persisted record requires schema_version 1")
        if record.get("status") not in {"success", "failed", "incomplete"}:
            raise EvaluationError("persisted record has invalid status")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise EvaluationError("persisted record requires sample_id")
        grouped.setdefault(sample_id, []).append(record)
    latest = {}
    for sample_id, attempts in grouped.items():
        successes = [record for record in attempts if record.get("status") == "success"]
        if len(successes) > 1:
            raise EvaluationError(f"duplicate successful record for {sample_id}")
        latest[sample_id] = successes[0] if successes else attempts[-1]
    return latest


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def evaluate_samples(
    samples: Sequence[EvaluationSample],
    adapter: EvaluationAdapter,
    answer_parser: Callable[[str], str | None],
    records_path: Path,
    bad_samples_path: Path,
) -> list[dict[str, Any]]:
    if not getattr(adapter, "method", ""):
        raise EvaluationError("adapter method must be nonempty")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise EvaluationError("input samples contain duplicate sample IDs")
    existing = latest_evaluation_records(records_path)
    completed = []
    for sample in samples:
        sample.validate()
        previous = existing.get(sample.sample_id)
        if previous is not None:
            if previous.get("prompt_fingerprint") != sample.prompt_fingerprint:
                raise EvaluationError(
                    f"prompt fingerprint changed for {sample.sample_id}"
                )
            if previous.get("method") != adapter.method:
                raise EvaluationError(
                    f"evaluation method changed for {sample.sample_id}"
                )
            if previous.get("status") == "success":
                completed.append(previous)
                continue
        base = _base_record(sample, adapter.method)
        try:
            generated = adapter.generate_one(sample)
            generated.validate()
            prediction = (
                answer_parser(generated.text)
                if sample.scoring_mode == "exact"
                else None
            )
            record = {
                **base,
                "status": "success",
                "prediction": prediction,
                "is_correct": (
                    prediction == sample.true_answer
                    if sample.scoring_mode == "exact"
                    else None
                ),
                "scoring_status": (
                    "scored" if sample.scoring_mode == "exact" else "external_required"
                ),
                "generation": {
                    "text": generated.text,
                    "token_ids": list(generated.token_ids),
                },
                "metrics": dict(generated.metrics),
                "diagnostics": dict(generated.diagnostics),
                "error": None,
            }
        except Exception as exc:
            record = {
                **base,
                "status": "failed",
                "prediction": None,
                "is_correct": None,
                "generation": None,
                "metrics": {},
                "diagnostics": {},
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            _append_jsonl(bad_samples_path, record)
        _append_jsonl(records_path, record)
        completed.append(record)
    return completed


def merge_evaluation_records(
    inputs: Sequence[Path], output: Path
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in inputs:
        for sample_id, record in latest_evaluation_records(path).items():
            if sample_id in merged:
                raise EvaluationError(f"duplicate sample across ranks: {sample_id}")
            merged[sample_id] = record
    ordered = sorted(
        merged.values(),
        key=lambda record: (
            str(record.get("subject", "")),
            int(record.get("question_index", -1)),
            str(record.get("sample_id", "")),
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in ordered
        ),
        encoding="utf-8",
    )
    partial.replace(output)
    return ordered


def summarize_evaluation_records(
    records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    successes = [record for record in records if record.get("status") == "success"]
    failures = [record for record in records if record.get("status") != "success"]
    if not successes:
        raise EvaluationError("cannot summarize records without successful samples")
    subjects: dict[str, list[bool]] = {}
    unscored_subjects: dict[str, int] = {}
    methods: dict[str, int] = {}
    numeric_metrics: dict[str, list[float]] = {}
    for record in successes:
        subject = str(record.get("subject", ""))
        is_correct = record.get("is_correct")
        if isinstance(is_correct, bool):
            subjects.setdefault(subject, []).append(is_correct)
        elif is_correct is None and record.get("scoring_mode") == "external":
            unscored_subjects[subject] = unscored_subjects.get(subject, 0) + 1
        else:
            raise EvaluationError(
                "successful record requires boolean is_correct or external scoring"
            )
        method = str(record.get("method", ""))
        methods[method] = methods.get(method, 0) + 1
        metrics = record.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise EvaluationError("successful record metrics must be an object")
        _validate_finite(metrics)
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_metrics.setdefault(str(name), []).append(float(value))
    scored = [
        record for record in successes if isinstance(record.get("is_correct"), bool)
    ]
    correct = sum(bool(record["is_correct"]) for record in scored)
    failure_reasons: dict[str, int] = {}
    for record in failures:
        error = record.get("error")
        error_type = error.get("type") if isinstance(error, Mapping) else None
        reason = str(error_type or record.get("status") or "unknown")
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    return {
        "schema_version": 1,
        "total_records": len(records),
        "successful_samples": len(successes),
        "failed_samples": len(failures),
        "scored_samples": len(scored),
        "unscored_samples": len(successes) - len(scored),
        "correct_samples": correct,
        "accuracy": None if not scored else correct / len(scored),
        "methods": dict(sorted(methods.items())),
        "metric_means": {
            name: sum(values) / len(values)
            for name, values in sorted(numeric_metrics.items())
        },
        "subjects": {
            subject: {
                "samples": len(values) + unscored_subjects.get(subject, 0),
                "scored": len(values),
                "unscored": unscored_subjects.get(subject, 0),
                "correct": sum(values),
                "accuracy": None if not values else sum(values) / len(values),
            }
            for subject, values in sorted(
                {
                    **{name: values for name, values in subjects.items()},
                    **{name: subjects.get(name, []) for name in unscored_subjects},
                }.items()
            )
        },
        "failed_sample_ids": sorted(
            str(record.get("sample_id")) for record in failures
        ),
        "failure_reasons": dict(sorted(failure_reasons.items())),
    }


def save_evaluation_summary(summary: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(output)
