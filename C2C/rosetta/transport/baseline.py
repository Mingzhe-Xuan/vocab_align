"""Reproducible baseline snapshots without model execution."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from .config import PENDING_CHECKPOINT, TransportConfig


class BaselineError(ValueError):
    """Raised when baseline inputs cannot form an auditable snapshot."""


@dataclass(frozen=True)
class BaselineSnapshot:
    schema_version: int
    input_fingerprint: str
    build_config: Dict[str, Any]
    seed: int
    code_version: str
    canonical_messages: list[Dict[str, str]]
    rendered_prompts: Dict[str, str]
    generation_config: Dict[str, Any]
    models: Dict[str, Any]
    checkpoint: Dict[str, Any]
    runtime: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def collect_runtime_info(packages: Sequence[str] = ("numpy", "torch", "transformers")) -> Dict[str, Any]:
    dependencies: Dict[str, str] = {}
    for package in packages:
        try:
            dependencies[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            dependencies[package] = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine() or "unavailable",
        "processor": platform.processor() or "unavailable",
        "cpu_count": os.cpu_count(),
        "gpu": {"status": "unavailable", "reason": "not-probed-by-baseline-snapshot"},
        "dependencies": dependencies,
    }


def _validate_messages(messages: Sequence[Mapping[str, Any]]) -> list[Dict[str, str]]:
    if not messages:
        raise BaselineError("canonical_messages cannot be empty")
    normalized = []
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise BaselineError(f"message {index} must contain only role and content")
        if not isinstance(message["role"], str) or not isinstance(message["content"], str):
            raise BaselineError(f"message {index} role/content must be strings")
        normalized.append({"role": message["role"], "content": message["content"]})
    return normalized


def _checkpoint_status(checkpoint: str | None) -> Dict[str, Any]:
    if checkpoint == PENDING_CHECKPOINT:
        return {"status": "pending", "value": checkpoint}
    if checkpoint is None:
        return {"status": "unavailable", "reason": "not-configured"}
    path = Path(checkpoint)
    if not path.exists():
        return {"status": "unavailable", "path": str(path), "reason": "missing"}
    return {"status": "available", "path": str(path.resolve())}


def freeze_baseline(
    config: TransportConfig,
    canonical_messages: Sequence[Mapping[str, Any]],
    rendered_prompts: Mapping[str, str],
    *,
    code_version: str,
    runtime: Mapping[str, Any] | None = None,
) -> BaselineSnapshot:
    config.validate()
    messages = _validate_messages(canonical_messages)
    if set(rendered_prompts) != {"source", "target"} or not all(
        isinstance(value, str) for value in rendered_prompts.values()
    ):
        raise BaselineError("rendered_prompts must contain source and target strings")
    if not code_version.strip():
        raise BaselineError("code_version is required")
    prompt_copy = {"source": rendered_prompts["source"], "target": rendered_prompts["target"]}
    fingerprint_payload = json.dumps(
        {"canonical_messages": messages, "rendered_prompts": prompt_copy},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return BaselineSnapshot(
        schema_version=1,
        input_fingerprint=hashlib.sha256(fingerprint_payload).hexdigest(),
        build_config=config.to_dict(),
        seed=config.seed,
        code_version=code_version,
        canonical_messages=messages,
        rendered_prompts=prompt_copy,
        generation_config=dict(config.generation),
        models={
            "source": {
                "name": config.source.name,
                "model_revision": config.source.revision,
                "tokenizer_revision": config.source.tokenizer_revision,
            },
            "target": {
                "name": config.target.name,
                "model_revision": config.target.revision,
                "tokenizer_revision": config.target.tokenizer_revision,
            },
        },
        checkpoint=_checkpoint_status(config.target.checkpoint),
        runtime=dict(runtime) if runtime is not None else collect_runtime_info(),
    )


def save_baseline(snapshot: BaselineSnapshot, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(snapshot.to_json() + "\n", encoding="utf-8")
