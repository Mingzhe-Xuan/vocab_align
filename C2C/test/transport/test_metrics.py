import pytest

from rosetta.transport.metrics import MetricsError, TransportMetrics


def test_metrics_validate_timing_and_cpu_memory_unavailable():
    metrics = TransportMetrics(0.1, 0.2, 0.3, 0.4, 1.0, 3, 3, 2, None)
    payload = metrics.to_dict()
    assert payload["memory_status"] == "unavailable"
    assert payload["peak_memory_bytes"] is None


def test_metrics_reject_inconsistent_total():
    with pytest.raises(MetricsError, match="do not sum"):
        TransportMetrics(0.1, 0.2, 0.3, 0.4, 2.0, 3, 3, 2, None).validate()
