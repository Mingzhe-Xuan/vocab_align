"""Deterministic paired statistics for unified transport evaluation records."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


class TransportStatisticsError(ValueError):
    """Raised when paired evaluation records cannot be compared safely."""


_LATENCY_FIELDS = {
    "source_prefill": "source_seconds",
    "transport": "transport_seconds",
    "receiver_prefill": "receiver_prefill_seconds",
    "decode": "decode_seconds",
}


def _index_records(
    records: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise TransportStatisticsError(f"{label} record requires sample_id")
        if sample_id in indexed:
            raise TransportStatisticsError(f"{label} has duplicate sample {sample_id}")
        indexed[sample_id] = record
    return indexed


def _paired_rows(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    list[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    dict[str, list[str]],
]:
    reference_by_id = _index_records(reference, "reference")
    candidate_by_id = _index_records(candidate, "candidate")
    excluded: dict[str, list[str]] = {
        "reference_not_success": [],
        "candidate_not_success": [],
        "reference_unscored": [],
        "candidate_unscored": [],
    }
    rows = []
    for sample_id in sorted(set(reference_by_id) & set(candidate_by_id)):
        reference_record = reference_by_id[sample_id]
        candidate_record = candidate_by_id[sample_id]
        excluded_row = False
        if reference_record.get("status") != "success":
            excluded["reference_not_success"].append(sample_id)
            excluded_row = True
        if candidate_record.get("status") != "success":
            excluded["candidate_not_success"].append(sample_id)
            excluded_row = True
        if excluded_row:
            continue
        if not isinstance(reference_record.get("is_correct"), bool):
            excluded["reference_unscored"].append(sample_id)
            excluded_row = True
        if not isinstance(candidate_record.get("is_correct"), bool):
            excluded["candidate_unscored"].append(sample_id)
            excluded_row = True
        if excluded_row:
            continue
        rows.append((sample_id, reference_record, candidate_record))
    return reference_by_id, candidate_by_id, rows, excluded


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _bootstrap(
    rows: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise TransportStatisticsError("bootstrap resamples must be a positive integer")
    if not isinstance(confidence, (int, float)) or not 0.0 < confidence < 1.0:
        raise TransportStatisticsError(
            "bootstrap confidence must be between zero and one"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise TransportStatisticsError("bootstrap seed must be a nonnegative integer")
    if not rows:
        return {
            "method": "paired_percentile_bootstrap",
            "resamples": resamples,
            "confidence": float(confidence),
            "seed": seed,
            "estimate": None,
            "lower": None,
            "upper": None,
            "status": "no_paired_scored_samples",
        }
    deltas = [
        int(bool(candidate["is_correct"])) - int(bool(reference["is_correct"]))
        for _, reference, candidate in rows
    ]
    sample_count = len(deltas)
    outcome_counts = np.bincount(np.asarray(deltas, dtype=np.int8) + 1, minlength=3)
    probabilities = outcome_counts / sample_count
    bootstrap_counts = np.random.default_rng(seed).multinomial(
        sample_count, probabilities, size=resamples
    )
    estimates = np.sort(
        (bootstrap_counts[:, 2] - bootstrap_counts[:, 0]) / sample_count
    ).tolist()
    tail = (1.0 - float(confidence)) / 2.0
    return {
        "method": "paired_percentile_bootstrap",
        "resamples": resamples,
        "confidence": float(confidence),
        "seed": seed,
        "estimate": sum(deltas) / sample_count,
        "lower": _quantile(estimates, tail),
        "upper": _quantile(estimates, 1.0 - tail),
        "status": "ok",
    }


def _mcnemar(
    rows: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    both_correct = reference_only = candidate_only = both_wrong = 0
    for _, reference, candidate in rows:
        reference_correct = bool(reference["is_correct"])
        candidate_correct = bool(candidate["is_correct"])
        if reference_correct and candidate_correct:
            both_correct += 1
        elif reference_correct:
            reference_only += 1
        elif candidate_correct:
            candidate_only += 1
        else:
            both_wrong += 1
    discordant = reference_only + candidate_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(reference_only, candidate_only)
        log_terms = [
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
            - discordant * math.log(2.0)
            for k in range(smaller + 1)
        ]
        maximum = max(log_terms)
        lower_tail = math.exp(maximum) * sum(
            math.exp(term - maximum) for term in log_terms
        )
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "method": "mcnemar_exact_two_sided",
        "both_correct": both_correct,
        "reference_only": reference_only,
        "candidate_only": candidate_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "p_value": p_value,
    }


def _label(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if value is None:
        metadata = record.get("prompt_metadata")
        if isinstance(metadata, Mapping):
            value = metadata.get(field)
    return "(unlabeled)" if value is None or value == "" else str(value)


def _slices(
    rows: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[bool, bool]]] = {}
    for sample_id, reference, candidate in rows:
        reference_label = _label(reference, field)
        candidate_label = _label(candidate, field)
        if reference_label != candidate_label:
            raise TransportStatisticsError(
                f"{field} differs for paired sample {sample_id}: "
                f"{reference_label!r} != {candidate_label!r}"
            )
        grouped.setdefault(reference_label, []).append(
            (bool(reference["is_correct"]), bool(candidate["is_correct"]))
        )
    return {
        label: {
            "samples": len(outcomes),
            "reference_correct": sum(reference for reference, _ in outcomes),
            "candidate_correct": sum(candidate for _, candidate in outcomes),
            "reference_accuracy": sum(reference for reference, _ in outcomes)
            / len(outcomes),
            "candidate_accuracy": sum(candidate for _, candidate in outcomes)
            / len(outcomes),
            "accuracy_delta": sum(
                candidate - reference for reference, candidate in outcomes
            )
            / len(outcomes),
        }
        for label, outcomes in sorted(grouped.items())
    }


def _latency(
    rows: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Any]]]
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    result: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for stage, metric_name in _LATENCY_FIELDS.items():
        result[stage] = {}
        for side, record_index in (("reference", 1), ("candidate", 2)):
            values = []
            for row in rows:
                metrics = row[record_index].get("metrics", {})
                if not isinstance(metrics, Mapping):
                    raise TransportStatisticsError(
                        f"{side} metrics for {row[0]} must be an object"
                    )
                value = metrics.get(metric_name)
                if value is None:
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or value < 0
                ):
                    raise TransportStatisticsError(
                        f"{side} {metric_name} for {row[0]} must be finite and nonnegative"
                    )
                values.append(float(value))
            result[stage][side] = {
                "count": len(values),
                "mean_seconds": None if not values else sum(values) / len(values),
            }
    return result


def _failure_index(
    reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    failures = []
    for side, records in (("reference", reference), ("candidate", candidate)):
        for record in records:
            if record.get("status") == "success":
                continue
            error = record.get("error")
            failures.append(
                {
                    "side": side,
                    "sample_id": str(record.get("sample_id", "")),
                    "status": str(record.get("status", "")),
                    "error_type": (
                        str(error.get("type", "")) if isinstance(error, Mapping) else ""
                    ),
                    "error_message": (
                        str(error.get("message", ""))
                        if isinstance(error, Mapping)
                        else ""
                    ),
                }
            )
    return sorted(
        failures,
        key=lambda item: (item["sample_id"], item["side"], item["status"]),
    )


def paired_transport_statistics(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_confidence: float = 0.95,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Summarize paired scored rows without silently treating missing data as zero."""

    reference_by_id, candidate_by_id, rows, excluded = _paired_rows(
        reference, candidate
    )
    paired_ids = [sample_id for sample_id, _, _ in rows]
    reference_correct = sum(bool(row[1]["is_correct"]) for row in rows)
    candidate_correct = sum(bool(row[2]["is_correct"]) for row in rows)
    paired_count = len(rows)
    subject_slices = _slices(rows, "subject")
    category_slices = _slices(rows, "category")
    if sum(item["samples"] for item in subject_slices.values()) != paired_count:
        raise TransportStatisticsError("subject slice counts are not conserved")
    if sum(item["samples"] for item in category_slices.values()) != paired_count:
        raise TransportStatisticsError("category slice counts are not conserved")
    return {
        "schema_version": 1,
        "complete_pairing": (
            set(reference_by_id) == set(candidate_by_id) and not any(excluded.values())
        ),
        "paired_samples": paired_count,
        "paired_sample_ids": paired_ids,
        "missing_candidate_ids": sorted(set(reference_by_id) - set(candidate_by_id)),
        "missing_reference_ids": sorted(set(candidate_by_id) - set(reference_by_id)),
        "excluded_ids": excluded,
        "reference_accuracy": (None if not rows else reference_correct / paired_count),
        "candidate_accuracy": (None if not rows else candidate_correct / paired_count),
        "accuracy_delta": (
            None if not rows else (candidate_correct - reference_correct) / paired_count
        ),
        "bootstrap": _bootstrap(
            rows,
            resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            seed=bootstrap_seed,
        ),
        "mcnemar": _mcnemar(rows),
        "slices": {"subject": subject_slices, "category": category_slices},
        "latency": _latency(rows),
        "failures": _failure_index(reference, candidate),
    }
