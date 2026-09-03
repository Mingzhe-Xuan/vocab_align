"""Narrow versioned evaluation loop for training-free transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset

from rosetta.transport.evaluation import (
    EvaluationSample,
    evaluate_samples,
    latest_evaluation_records,
    save_evaluation_summary,
    summarize_evaluation_records,
)
from script.evaluation.transport_adapter import (
    create_training_free_transport_adapter,
)


def _paths(evaluator: Any) -> tuple[Path, Path, Path]:
    output = evaluator.output_config
    root = Path(output["output_dir"])
    return (
        Path(output.get("records", root / "stt.records.jsonl")),
        Path(output.get("bad_samples", root / "stt.bad-samples.jsonl")),
        Path(output.get("summary", root / "stt.summary.json")),
    )


def _load_subject_dataset(evaluator: Any, subject: str) -> Any:
    name = evaluator.dataset_config["dataset_name"]
    if evaluator.dataset_name in {"math-500", "openbookqa", "mmlu-pro"}:
        return load_dataset(name)
    if evaluator.dataset_name == "gsm8k":
        return load_dataset(name, "main")
    return load_dataset(name, subject)


def _subject_samples(
    evaluator: Any, subject: str, source_tokenizer: Any | None = None
) -> list[EvaluationSample]:
    dataset = _load_subject_dataset(evaluator, subject)
    test_data = dataset[evaluator.dataset_config["test_split"]]
    interval = evaluator.eval_config.get("sample_interval", 1)
    indices = list(range(0, len(test_data), interval))
    limit = evaluator.eval_config.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        indices = indices[:limit]
    elif isinstance(limit, (list, tuple)) and len(limit) == 2:
        start = 0 if limit[0] is None else int(limit[0])
        end = len(test_data) if limit[1] is None else int(limit[1])
        indices = [index for index in indices if start <= index < end]

    samples = []
    for index in indices:
        example = test_data[index]
        scoring_mode = "exact"
        if evaluator.dataset_name == "longbench":
            if source_tokenizer is None:
                raise ValueError(
                    "LongBench prompt formatting requires source tokenizer"
                )
            evaluator.current_evaluating_subject = subject
            prompt = evaluator._format_longbench_example(example, source_tokenizer)
            true_answer = "<external-longbench-scorer>"
            scoring_mode = "external"
        else:
            true_answer = evaluator.parse_answer(example)
            if true_answer is None:
                continue
            prompt = evaluator.format_example(
                example,
                use_cot=evaluator.eval_config["use_cot"],
                use_template=evaluator.eval_config["use_template"],
            )
        metadata = {
            "dataset": evaluator.dataset_name,
            "use_cot": evaluator.eval_config["use_cot"],
            "use_template": evaluator.eval_config["use_template"],
            "answer_method": evaluator.eval_config["answer_method"],
        }
        if scoring_mode == "external":
            metadata["longbench"] = {
                "answers": example.get("answers", []),
                "all_classes": example.get("all_classes", []),
                "length": example.get("length"),
                "id": example.get("_id"),
            }
        samples.append(
            EvaluationSample(
                sample_id=f"{evaluator.dataset_name}:{subject}:{index}",
                subject=subject,
                question_index=index,
                canonical_messages=[{"role": "user", "content": prompt}],
                prompt=prompt,
                true_answer=true_answer,
                prompt_metadata=metadata,
                scoring_mode=scoring_mode,
            )
        )
    return samples


def run_training_free_transport_evaluation(evaluator: Any) -> None:
    if evaluator.eval_config["answer_method"] != "generate":
        raise ValueError(
            "training_free_transport currently requires answer_method=generate"
        )
    gpu_ids = evaluator.eval_config["gpu_ids"]
    if len(gpu_ids) != 1:
        raise ValueError(
            "training_free_transport evaluator currently requires exactly one GPU"
        )
    torch.cuda.set_device(gpu_ids[0])
    adapter = create_training_free_transport_adapter(evaluator.model_config)
    records_path, bad_samples_path, summary_path = _paths(evaluator)
    subjects = evaluator.dataset_config["subjects"]
    requested = evaluator.eval_config.get("subjects")
    if requested is not None:
        subjects = [subject for subject in subjects if subject in requested]
    for subject in subjects:
        evaluate_samples(
            _subject_samples(
                evaluator, subject, getattr(adapter, "source_tokenizer", None)
            ),
            adapter,
            evaluator.extract_predicted_answer,
            records_path,
            bad_samples_path,
        )
    records = list(latest_evaluation_records(records_path).values())
    summary = summarize_evaluation_records(records)
    summary["dataset"] = evaluator.dataset_name
    save_evaluation_summary(summary, summary_path)
    print(f"Transport records saved to {records_path}")
    print(f"Transport summary saved to {summary_path}")
