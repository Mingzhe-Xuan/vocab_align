"""Validated, reproducible configuration for vocabulary transport."""

from __future__ import annotations

import json
import math
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
    tokenizer_fingerprint: Optional[str] = None
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
        if self.tokenizer_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.tokenizer_fingerprint
        ):
            raise ConfigError("tokenizer fingerprint must be a 64-character SHA-256")
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
                "tokenizer_fingerprint",
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
class TransportConstructionSpec:
    """Numerical settings used to build a full-vocabulary transport artifact."""

    epsilon: float = 0.5
    tolerance: float = 1e-9
    max_iter: int = 10_000
    smoothing: float = 1e-8

    def validate(self) -> None:
        for field_name in ("epsilon", "tolerance", "smoothing"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ConfigError(
                    f"construction {field_name} must be finite and positive"
                )
        if (
            isinstance(self.max_iter, bool)
            or not isinstance(self.max_iter, int)
            or self.max_iter <= 0
        ):
            raise ConfigError("construction max_iter must be a positive integer")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransportConstructionSpec":
        _only_fields(
            payload,
            {"epsilon", "tolerance", "max_iter", "smoothing"},
            "construction",
        )
        try:
            value = cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid construction configuration: {exc}") from exc
        value.validate()
        return value


@dataclass(frozen=True)
class SpecialTokenPolicy:
    """Frozen safe support policy implemented by the formal builder."""

    source_support: str = "full_tokenizer"
    target_support: str = "ordinary_only"
    mapping: str = "exact_kind_then_literal_bytes"
    receiver_boundary: str = "native"

    def validate(self) -> None:
        expected = SpecialTokenPolicy()
        if self != expected:
            raise ConfigError(
                "special token policy must use full source support, ordinary-only "
                "target support, exact-kind/literal-byte mapping, and native receiver "
                "boundaries"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpecialTokenPolicy":
        _only_fields(
            payload,
            {"source_support", "target_support", "mapping", "receiver_boundary"},
            "special token policy",
        )
        try:
            value = cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid special token policy: {exc}") from exc
        value.validate()
        return value


@dataclass(frozen=True)
class CollaborationSpec:
    """Frozen roles and prompt semantics for Planner-to-Thinker STT."""

    sender_role: str = "planner"
    receiver_role: str = "thinker"
    sender_system_prompt: str = (
        "You are the planner. Analyze the problem step by step and produce a useful "
        "reasoning plan for another model."
    )
    receiver_system_prompt: str = (
        "You are the thinker. Use the preceding planner context, think through the "
        "problem step by step yourself, and give the final answer."
    )
    sender_enable_thinking: bool = True
    receiver_enable_thinking: bool = True
    context_mode: str = "sender_prompt_and_think"
    receiver_problem_mode: str = "explicit"

    def validate(self) -> None:
        if self.sender_role != "planner" or self.receiver_role != "thinker":
            raise ConfigError("collaboration roles must be planner and thinker")
        if (
            not self.sender_system_prompt.strip()
            or not self.receiver_system_prompt.strip()
        ):
            raise ConfigError("collaboration system prompts must be nonempty")
        if self.sender_enable_thinking is not True:
            raise ConfigError("sender thinking must be explicitly enabled")
        if self.receiver_enable_thinking is not True:
            raise ConfigError("receiver thinking must be explicitly enabled")
        if self.context_mode != "sender_prompt_and_think":
            raise ConfigError("collaboration must transport sender prompt and think")
        if self.receiver_problem_mode != "explicit":
            raise ConfigError("receiver must receive the problem explicitly")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CollaborationSpec":
        _only_fields(
            payload,
            {
                "sender_role",
                "receiver_role",
                "sender_system_prompt",
                "receiver_system_prompt",
                "sender_enable_thinking",
                "receiver_enable_thinking",
                "context_mode",
                "receiver_problem_mode",
            },
            "collaboration",
        )
        try:
            value = cls(**payload)
        except TypeError as exc:
            raise ConfigError(f"invalid collaboration configuration: {exc}") from exc
        value.validate()
        return value


def _validate_generation(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("do_sample") is not False:
        raise ConfigError(f"{label} generation must explicitly use greedy decoding")
    max_new_tokens = payload.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise ConfigError(f"{label} max_new_tokens must be a positive integer")
    temperature = payload.get("temperature", 1.0)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0
    ):
        raise ConfigError(f"{label} temperature must be finite and positive")


@dataclass(frozen=True)
class TransportConfig:
    schema_version: int
    source: ModelSpec
    target: ModelSpec
    data: DataSpec
    seed: int
    output_path: str
    output_schema: str
    expected_artifact_shape: Optional[Tuple[int, int]] = None
    construction: TransportConstructionSpec = field(
        default_factory=TransportConstructionSpec
    )
    transport: TransportInferenceSpec = field(default_factory=TransportInferenceSpec)
    special_tokens: SpecialTokenPolicy = field(default_factory=SpecialTokenPolicy)
    collaboration: CollaborationSpec = field(default_factory=CollaborationSpec)
    sender_generation: Dict[str, Any] = field(
        default_factory=lambda: {
            "do_sample": False,
            "max_new_tokens": 128,
            "temperature": 1.0,
        }
    )
    generation: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ConfigError("unsupported configuration schema_version")
        self.source.validate()
        self.target.validate()
        self.data.validate()
        self.construction.validate()
        self.transport.validate()
        self.special_tokens.validate()
        self.collaboration.validate()
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
        if self.expected_artifact_shape is not None and (
            len(self.expected_artifact_shape) != 2
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in self.expected_artifact_shape
            )
        ):
            raise ConfigError(
                "expected_artifact_shape must be two positive integer dimensions"
            )
        if not isinstance(self.generation, dict):
            raise ConfigError("generation must be an object")
        if not isinstance(self.sender_generation, dict):
            raise ConfigError("sender_generation must be an object")
        _validate_generation(self.sender_generation, "sender")
        _validate_generation(self.generation, "receiver")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["data"]["build_splits"] = list(self.data.build_splits)
        if self.expected_artifact_shape is not None:
            payload["expected_artifact_shape"] = list(self.expected_artifact_shape)
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
                "expected_artifact_shape",
                "construction",
                "transport",
                "special_tokens",
                "collaboration",
                "sender_generation",
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
                expected_artifact_shape=(
                    tuple(payload["expected_artifact_shape"])
                    if payload.get("expected_artifact_shape") is not None
                    else None
                ),
                construction=TransportConstructionSpec.from_dict(
                    payload.get("construction", {})
                ),
                transport=TransportInferenceSpec.from_dict(
                    payload.get("transport", {})
                ),
                special_tokens=SpecialTokenPolicy.from_dict(
                    payload.get("special_tokens", {})
                ),
                collaboration=CollaborationSpec.from_dict(
                    payload.get("collaboration", {})
                ),
                sender_generation=dict(
                    payload.get(
                        "sender_generation",
                        {
                            "do_sample": False,
                            "max_new_tokens": 128,
                            "temperature": 1.0,
                        },
                    )
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
