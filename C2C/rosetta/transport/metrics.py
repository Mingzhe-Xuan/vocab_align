"""Unified stage latency, length, and memory records for STT evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


class MetricsError(ValueError):
    """Raised when a metrics record is internally inconsistent."""


@dataclass(frozen=True)
class TransportMetrics:
    source_seconds: float
    transport_seconds: float
    receiver_prefill_seconds: float
    decode_seconds: float
    total_seconds: float
    source_input_tokens: int
    virtual_tokens: int
    output_tokens: int
    peak_memory_bytes: int | None
    receiver_prompt_tokens: int = 0

    def validate(self, timing_tolerance: float = 1e-6) -> None:
        stages = (
            self.source_seconds,
            self.transport_seconds,
            self.receiver_prefill_seconds,
            self.decode_seconds,
        )
        if any(value < 0 for value in stages) or self.total_seconds < 0:
            raise MetricsError("latencies must be nonnegative")
        if abs(sum(stages) - self.total_seconds) > timing_tolerance:
            raise MetricsError("stage latencies do not sum to total_seconds")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.source_input_tokens,
                self.virtual_tokens,
                self.output_tokens,
                self.receiver_prompt_tokens,
            )
        ):
            raise MetricsError("token lengths must be nonnegative integers")
        if self.peak_memory_bytes is not None and self.peak_memory_bytes < 0:
            raise MetricsError("peak memory must be unavailable or nonnegative")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["memory_status"] = (
            "unavailable" if self.peak_memory_bytes is None else "available"
        )
        payload["prefill_tokens"] = self.virtual_tokens + self.receiver_prompt_tokens
        return payload
