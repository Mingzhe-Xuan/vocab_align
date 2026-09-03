"""Pre-registered ablation expansion and paired evaluation statistics."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class AblationError(ValueError):
    """Raised when an ablation plan or paired result set is inconsistent."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise AblationError("ablation values must be JSON serializable") from exc


def _validate_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AblationError(f"{path} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise AblationError(f"{path} object keys must be nonempty strings")
            _validate_value(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_value(item, f"{path}[{index}]")
        return
    raise AblationError(f"{path} contains unsupported {type(value).__name__}")


@dataclass(frozen=True)
class AblationRun:
    phase: str
    split: str
    parameters: Mapping[str, Any]
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "split": self.split,
            "parameters": dict(self.parameters),
            "run_id": self.run_id,
        }


@dataclass(frozen=True)
class AblationPlan:
    schema_version: int
    dev_split: str
    test_split: str
    dimensions: Mapping[str, Sequence[Any]]
    frozen: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AblationPlan":
        allowed = {
            "schema_version",
            "dev_split",
            "test_split",
            "dimensions",
            "frozen",
        }
        unknown = set(payload).difference(allowed)
        if unknown:
            raise AblationError(f"unknown ablation plan fields: {sorted(unknown)}")
        try:
            plan = cls(
                schema_version=payload["schema_version"],
                dev_split=payload["dev_split"],
                test_split=payload["test_split"],
                dimensions=payload["dimensions"],
                frozen=payload["frozen"],
            )
        except KeyError as exc:
            raise AblationError(f"missing ablation field: {exc.args[0]}") from exc
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.schema_version != 1:
            raise AblationError("unsupported ablation schema_version")
        if not isinstance(self.dev_split, str) or not self.dev_split:
            raise AblationError("dev_split must be a nonempty string")
        if not isinstance(self.test_split, str) or not self.test_split:
            raise AblationError("test_split must be a nonempty string")
        if "test" in self.dev_split.lower():
            raise AblationError("dev_split cannot be a benchmark test split")
        if not isinstance(self.dimensions, Mapping) or not self.dimensions:
            raise AblationError("dimensions must be a nonempty object")
        for name, choices in self.dimensions.items():
            if not isinstance(name, str) or not name:
                raise AblationError("dimension names must be nonempty strings")
            if isinstance(choices, (str, bytes)) or not isinstance(choices, Sequence):
                raise AblationError(f"dimension {name} choices must be a sequence")
            if not choices:
                raise AblationError(f"dimension {name} must have at least one choice")
            serialized = []
            for index, choice in enumerate(choices):
                _validate_value(choice, f"dimensions.{name}[{index}]")
                serialized.append(_canonical(choice))
            if len(serialized) != len(set(serialized)):
                raise AblationError(f"dimension {name} contains duplicate choices")
        if set(self.frozen) != set(self.dimensions):
            raise AblationError("frozen parameters must exactly cover all dimensions")
        for name, value in self.frozen.items():
            _validate_value(value, f"frozen.{name}")
            allowed = {_canonical(choice) for choice in self.dimensions[name]}
            if _canonical(value) not in allowed:
                raise AblationError(f"frozen {name} is outside its dev search space")

    def expand(self, phase: str) -> list[AblationRun]:
        self.validate()
        if phase not in {"dev", "test"}:
            raise AblationError("phase must be dev or test")
        if phase == "test":
            combinations = [dict(self.frozen)]
            split = self.test_split
        else:
            names = sorted(self.dimensions)
            combinations = [
                dict(zip(names, values))
                for values in itertools.product(
                    *(self.dimensions[name] for name in names)
                )
            ]
            split = self.dev_split
        runs = []
        for parameters in combinations:
            payload = {"phase": phase, "split": split, "parameters": parameters}
            run_id = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()[
                :16
            ]
            runs.append(AblationRun(phase, split, parameters, run_id))
        if len({run.run_id for run in runs}) != len(runs):
            raise AblationError("ablation expansion produced duplicate run IDs")
        return runs


def _successful_by_id(
    records: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    indexed = {}
    for record in records:
        if record.get("status") != "success":
            continue
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise AblationError(f"{label} successful record requires sample_id")
        if sample_id in indexed:
            raise AblationError(f"{label} has duplicate successful sample {sample_id}")
        if not isinstance(record.get("is_correct"), bool):
            raise AblationError(f"{label} successful record requires is_correct")
        indexed[sample_id] = record
    return indexed


def paired_evaluation_summary(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_by_id = _successful_by_id(reference, "reference")
    candidate_by_id = _successful_by_id(candidate, "candidate")
    reference_ids = set(reference_by_id)
    candidate_ids = set(candidate_by_id)
    paired_ids = sorted(reference_ids & candidate_ids)
    missing_candidate = sorted(reference_ids - candidate_ids)
    missing_reference = sorted(candidate_ids - reference_ids)
    if paired_ids:
        reference_correct = sum(
            bool(reference_by_id[sample_id]["is_correct"]) for sample_id in paired_ids
        )
        candidate_correct = sum(
            bool(candidate_by_id[sample_id]["is_correct"]) for sample_id in paired_ids
        )
        reference_accuracy = reference_correct / len(paired_ids)
        candidate_accuracy = candidate_correct / len(paired_ids)
        candidate_wins = sum(
            not reference_by_id[sample_id]["is_correct"]
            and candidate_by_id[sample_id]["is_correct"]
            for sample_id in paired_ids
        )
        reference_wins = sum(
            reference_by_id[sample_id]["is_correct"]
            and not candidate_by_id[sample_id]["is_correct"]
            for sample_id in paired_ids
        )
    else:
        reference_accuracy = None
        candidate_accuracy = None
        candidate_wins = 0
        reference_wins = 0
    return {
        "schema_version": 1,
        "complete_pairing": not missing_candidate and not missing_reference,
        "paired_samples": len(paired_ids),
        "paired_sample_ids": paired_ids,
        "missing_candidate_ids": missing_candidate,
        "missing_reference_ids": missing_reference,
        "reference_accuracy": reference_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": (
            None
            if reference_accuracy is None
            else candidate_accuracy - reference_accuracy
        ),
        "candidate_wins": candidate_wins,
        "reference_wins": reference_wins,
    }
