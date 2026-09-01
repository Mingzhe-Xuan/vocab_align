"""Validated, reproducible configuration for vocabulary transport."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple


PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
PENDING_CHECKPOINT = "pending-new-projector-training"


class ConfigError(ValueError):
    """Raised when a transport configuration is incomplete or unsafe."""


def _only_fields(payload: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ConfigError(f"unknown {context} fields: {sorted(unknown)}")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    revision: str
    tokenizer_revision: str
    dtype: str = "bfloat16"
    device_map: Any = "auto"
    checkpoint: Optional[str] = None

    def validate(self) -> None:
        if not self.name.strip():
            raise ConfigError("model name is required")
        for label, revision in (
            ("model revision", self.revision),
            ("tokenizer revision", self.tokenizer_revision),
        ):
            if not PINNED_REVISION.fullmatch(revision):
                raise ConfigError(f"{label} must be a pinned 40-character commit SHA")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ConfigError(f"unsupported dtype: {self.dtype}")
        if self.checkpoint is not None and not str(self.checkpoint).strip():
            raise ConfigError("checkpoint cannot be empty")

    @property
    def checkpoint_available(self) -> bool:
        return self.checkpoint not in {None, PENDING_CHECKPOINT}

    def require_checkpoint(self) -> Path:
        if not self.checkpoint_available:
            raise ConfigError("checkpoint is pending or unavailable")
        return Path(str(self.checkpoint))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelSpec":
        _only_fields(
            payload,
            {
                "name",
                "revision",
                "tokenizer_revision",
                "dtype",
                "device_map",
                "checkpoint",
            },
            "model",
        )
        try:
            value = cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid model configuration: {exc}") from exc
        value.validate()
        return value


@dataclass(frozen=True)
class DataSpec:
    dataset: str
    revision: str
    build_splits: Tuple[str, ...] = ("transport_train",)
    dev_fraction: float = 0.01

    def validate(self) -> None:
        if not self.dataset.strip():
            raise ConfigError("dataset name is required")
        if not PINNED_REVISION.fullmatch(self.revision):
            raise ConfigError(
                "dataset revision must be a pinned 40-character commit SHA"
            )
        if not self.build_splits:
            raise ConfigError("at least one transport build split is required")
        if any("test" in split.lower() for split in self.build_splits):
            raise ConfigError("benchmark test splits cannot build vocabulary transport")
        if not 0 < self.dev_fraction < 1:
            raise ConfigError("dev_fraction must be between zero and one")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataSpec":
        _only_fields(
            payload, {"dataset", "revision", "build_splits", "dev_fraction"}, "data"
        )
        normalized = dict(payload)
        if "build_splits" in normalized:
            normalized["build_splits"] = tuple(normalized["build_splits"])
        try:
            value = cls(**normalized)
        except TypeError as exc:
            raise ConfigError(f"invalid data configuration: {exc}") from exc
        value.validate()
        return value


@dataclass(frozen=True)
class TransportInferenceSpec:
    tau: float = 1.0
    causal_shift: bool = True
    source_top_m: Optional[int] = None

    def validate(self) -> None:
        if not isinstance(self.tau, (int, float)) or isinstance(self.tau, bool):
            raise ConfigError("transport tau must be numeric")
        if not 0 < float(self.tau) < float("inf"):
            raise ConfigError("transport tau must be finite and positive")
        if not isinstance(self.causal_shift, bool):
            raise ConfigError("causal_shift must be boolean")
        if self.source_top_m is not None and (
            isinstance(self.source_top_m, bool)
            or not isinstance(self.source_top_m, int)
            or self.source_top_m <= 0
        ):
            raise ConfigError("source_top_m must be null or a positive integer")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportInferenceSpec":
        _only_fields(payload, {"tau", "causal_shift", "source_top_m"}, "transport")
        try:
            value = cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid transport configuration: {exc}") from exc
        value.validate()
        return value


@dataclass(frozen=True)
class TransportConfig:
    schema_version: int
    source: ModelSpec
    target: ModelSpec
    data: DataSpec
    seed: int
    output_path: str
    output_schema: str
    transport: TransportInferenceSpec = field(default_factory=TransportInferenceSpec)
    generation: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigError("unsupported configuration schema_version")
        self.source.validate()
        self.target.validate()
        self.data.validate()
        self.transport.validate()
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ConfigError("seed must be a nonnegative integer")
        if not str(self.output_path).strip():
            raise ConfigError("output_path is required")
        if not self.output_schema.strip():
            raise ConfigError("output_schema is required")
        if not isinstance(self.generation, dict):
            raise ConfigError("generation must be an object")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["data"]["build_splits"] = list(self.data.build_splits)
        return payload

    def to_json(self) -> str:
        self.validate()
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportConfig":
        _only_fields(
            payload,
            {
                "schema_version",
                "source",
                "target",
                "data",
                "seed",
                "output_path",
                "output_schema",
                "transport",
                "generation",
            },
            "transport",
        )
        try:
            value = cls(
                schema_version=payload["schema_version"],
                source=ModelSpec.from_dict(payload["source"]),
                target=ModelSpec.from_dict(payload["target"]),
                data=DataSpec.from_dict(payload["data"]),
                seed=payload["seed"],
                output_path=payload["output_path"],
                output_schema=payload["output_schema"],
                transport=TransportInferenceSpec.from_dict(
                    payload.get("transport", {})
                ),
                generation=dict(payload.get("generation", {})),
            )
        except KeyError as exc:
            raise ConfigError(
                f"missing required configuration field: {exc.args[0]}"
            ) from exc
        value.validate()
        return value

    @classmethod
    def from_json(cls, payload: str) -> "TransportConfig":
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid configuration JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ConfigError("configuration root must be an object")
        return cls.from_dict(decoded)
