import pytest

from rosetta.transport.statistics import (
    TransportStatisticsError,
    paired_transport_statistics,
)


def _record(
    sample_id,
    correct,
    *,
    subject="math",
    category="reasoning",
    status="success",
    source=1.0,
):
    return {
        "sample_id": sample_id,
        "status": status,
        "is_correct": correct,
        "subject": subject,
        "prompt_metadata": {"category": category},
        "metrics": {
            "source_seconds": source,
            "transport_seconds": 0.2,
            "receiver_prefill_seconds": 0.3,
            "decode_seconds": 0.4,
        },
        "error": (
            None
            if status == "success"
            else {"type": "RuntimeError", "message": "synthetic failure"}
        ),
    }


def test_bootstrap_is_reproducible_and_interval_is_ordered():
    reference = [_record(str(i), i % 3 == 0) for i in range(12)]
    candidate = [_record(str(i), i % 2 == 0) for i in range(12)]
    first = paired_transport_statistics(
        reference, candidate, bootstrap_resamples=500, bootstrap_seed=17
    )
    second = paired_transport_statistics(
        reference, candidate, bootstrap_resamples=500, bootstrap_seed=17
    )
    assert first["bootstrap"] == second["bootstrap"]
    interval = first["bootstrap"]
    assert interval["lower"] <= interval["estimate"] <= interval["upper"]


def test_mcnemar_counts_handmade_table_and_all_identical():
    reference = [
        _record("both-correct", True),
        _record("reference-only", True),
        _record("candidate-only-1", False),
        _record("candidate-only-2", False),
        _record("both-wrong", False),
    ]
    candidate = [
        _record("both-correct", True),
        _record("reference-only", False),
        _record("candidate-only-1", True),
        _record("candidate-only-2", True),
        _record("both-wrong", False),
    ]
    result = paired_transport_statistics(reference, candidate)["mcnemar"]
    assert result == {
        "method": "mcnemar_exact_two_sided",
        "both_correct": 1,
        "reference_only": 1,
        "candidate_only": 2,
        "both_wrong": 1,
        "discordant": 3,
        "p_value": 1.0,
    }

    identical = paired_transport_statistics(reference, reference)["mcnemar"]
    assert identical["discordant"] == 0
    assert identical["p_value"] == 1.0


def test_slices_conserve_pairs_and_latency_keeps_stages_distinct():
    reference = [
        _record("a", True, subject="algebra", category="stem", source=1.0),
        _record("b", False, subject="history", category="humanities", source=3.0),
    ]
    candidate = [
        _record("a", False, subject="algebra", category="stem", source=2.0),
        _record("b", True, subject="history", category="humanities", source=4.0),
    ]
    result = paired_transport_statistics(reference, candidate)
    assert sum(x["samples"] for x in result["slices"]["subject"].values()) == 2
    assert sum(x["samples"] for x in result["slices"]["category"].values()) == 2
    assert set(result["latency"]) == {
        "source_prefill",
        "transport",
        "receiver_prefill",
        "decode",
    }
    assert result["latency"]["source_prefill"]["reference"]["mean_seconds"] == 2.0
    assert result["latency"]["source_prefill"]["candidate"]["mean_seconds"] == 3.0
    assert result["latency"]["transport"]["reference"]["mean_seconds"] == 0.2


def test_missing_unscored_and_failed_rows_are_explicit_and_indexed():
    reference = [
        _record("paired", True),
        _record("failed", None, status="failed"),
        _record("missing-candidate", True),
        _record("unscored", None),
    ]
    candidate = [
        _record("paired", False),
        _record("failed", True),
        _record("missing-reference", True),
        _record("unscored", True),
    ]
    result = paired_transport_statistics(reference, candidate)
    assert result["complete_pairing"] is False
    assert result["paired_sample_ids"] == ["paired"]
    assert result["missing_candidate_ids"] == ["missing-candidate"]
    assert result["missing_reference_ids"] == ["missing-reference"]
    assert result["excluded_ids"]["reference_not_success"] == ["failed"]
    assert result["excluded_ids"]["reference_unscored"] == ["unscored"]
    assert result["failures"] == [
        {
            "side": "reference",
            "sample_id": "failed",
            "status": "failed",
            "error_type": "RuntimeError",
            "error_message": "synthetic failure",
        }
    ]

    no_pairs = paired_transport_statistics(
        [_record("external", None)], [_record("external", None)]
    )
    assert no_pairs["bootstrap"]["status"] == "no_paired_scored_samples"
    assert no_pairs["mcnemar"]["p_value"] == 1.0


def test_missing_latency_is_not_imputed_as_zero():
    reference = _record("a", True)
    candidate = _record("a", True)
    del candidate["metrics"]["transport_seconds"]
    result = paired_transport_statistics([reference], [candidate])
    assert result["latency"]["transport"]["candidate"] == {
        "count": 0,
        "mean_seconds": None,
    }


def test_invalid_statistics_inputs_and_label_drift_fail():
    record = _record("a", True)
    with pytest.raises(TransportStatisticsError, match="duplicate"):
        paired_transport_statistics([record, record], [record])
    with pytest.raises(TransportStatisticsError, match="resamples"):
        paired_transport_statistics([record], [record], bootstrap_resamples=0)
    with pytest.raises(TransportStatisticsError, match="category differs"):
        paired_transport_statistics(
            [record], [_record("a", True, category="different")]
        )
