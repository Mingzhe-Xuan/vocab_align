"""Run one reproducible STT prompt and atomically save diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch
import yaml

from rosetta.transport.artifact import load_transport_artifact
from rosetta.transport.config import TransportConfig
from rosetta.transport.token_metadata import tokenizer_fingerprint
from rosetta.transport.wrapper import (
    TrainingFreeTransportModel,
    TransportGenerationOutput,
)


class SmokeError(ValueError):
    """Raised when smoke inputs or structured outputs are incomplete."""


RUNTIME_PROFILES = {
    "project-cu124": {
        "torch": "2.6.0",
        "accelerate": "1.9.0",
        "transformers": "4.52.4",
    },
    "blackwell-cu128": {
        "torch": "2.7.1+cu128",
        "accelerate": "1.9.0",
        "transformers": "4.52.4",
    },
}


def _git_version() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _tensor_summary(
    value: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> dict[str, float]:
    selected = value.detach().float().cpu()
    if attention_mask is not None:
        mask = attention_mask.detach().bool().cpu()
        if selected.shape != mask.shape:
            raise SmokeError("transport statistic and attention mask shapes differ")
        selected = selected[mask]
    flat = selected.flatten()
    if flat.numel() == 0:
        raise SmokeError("transport statistic cannot be empty")
    return {
        "minimum": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "maximum": float(flat.max().item()),
    }


def _source_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _validate_generation(generation: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "max_new_tokens",
        "eos_token_id",
        "pad_token_id",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
    }
    unknown = set(generation).difference(allowed)
    if unknown:
        raise SmokeError(f"unsupported generation fields: {sorted(unknown)}")
    max_new_tokens = generation.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= 2
    ):
        raise SmokeError("smoke max_new_tokens must be 1 or 2")
    return dict(generation)


def validate_runtime_requirements(
    artifact_path: Path,
    output_path: Path,
    *,
    require_cuda: bool,
    require_locked_runtime: bool,
    min_gpu_memory_gib: float,
    runtime_profile: str = "project-cu124",
) -> None:
    if not isinstance(min_gpu_memory_gib, (int, float)) or not (
        0 < float(min_gpu_memory_gib) < float("inf")
    ):
        raise SmokeError("minimum GPU memory must be finite and positive")
    if not artifact_path.is_file():
        raise SmokeError(f"transport artifact does not exist: {artifact_path}")
    partial = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or partial.exists():
        raise SmokeError(f"refusing to overwrite smoke output: {output_path}")
    if require_locked_runtime:
        if runtime_profile not in RUNTIME_PROFILES:
            raise SmokeError(f"unknown runtime profile: {runtime_profile}")
        mismatches = []
        for package, expected in RUNTIME_PROFILES[runtime_profile].items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                actual = "missing"
            if actual != expected:
                mismatches.append(f"{package}={actual} (expected {expected})")
        if mismatches:
            raise SmokeError("locked runtime mismatch: " + ", ".join(mismatches))
    if not require_cuda:
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SmokeError("CUDA GPU is required for the real-model smoke")
    available_gib = max(
        torch.cuda.get_device_properties(index).total_memory / (1024**3)
        for index in range(torch.cuda.device_count())
    )
    if available_gib < min_gpu_memory_gib:
        raise SmokeError(
            f"largest visible GPU has {available_gib:.1f} GiB; "
            f"at least {min_gpu_memory_gib:.1f} GiB is required"
        )
    compiled_arches = set(torch.cuda.get_arch_list())
    unsupported = []
    for index in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(index)
        architecture = f"sm_{major}{minor}"
        if architecture not in compiled_arches:
            unsupported.append(f"cuda:{index}={architecture}")
    if unsupported:
        raise SmokeError(
            "PyTorch wheel lacks visible GPU architectures "
            f"{unsupported}; compiled arches are {sorted(compiled_arches)}"
        )


def runtime_metadata(runtime_profile: str) -> dict[str, Any]:
    packages = {}
    for package in RUNTIME_PROFILES["project-cu124"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    return {
        "python": platform.python_version(),
        "profile": runtime_profile,
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "compiled_cuda_arches": (
            torch.cuda.get_arch_list() if torch.cuda.is_available() else []
        ),
        "cuda_devices": cuda_devices,
    }


def _receiver_only_report(
    wrapper: TrainingFreeTransportModel,
    target_tokenizer: Any,
    prompt: str,
    generation: Mapping[str, Any],
) -> dict[str, Any]:
    encoded = target_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = torch.as_tensor(
        encoded.get("attention_mask", torch.ones_like(input_ids)), dtype=torch.long
    )
    if attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    device = wrapper.receiver_model.get_input_embeddings().weight.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = perf_counter()
    sequences = wrapper.generate(
        receiver_input_ids=input_ids.to(device),
        receiver_attention_mask=attention_mask.to(device),
        transport=False,
        **dict(generation),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    if not isinstance(sequences, torch.Tensor) or sequences.ndim != 2:
        raise SmokeError("receiver-only generation must return rank-two token IDs")
    if sequences.shape[1] < input_ids.shape[1]:
        raise SmokeError("receiver-only output is shorter than its input")
    generated = sequences[:, input_ids.shape[1] :].detach().cpu().tolist()
    return {
        "shapes": {
            "receiver_input_ids": list(input_ids.shape),
            "receiver_full_sequence_ids": list(sequences.shape),
        },
        "metrics": {
            "total_seconds": elapsed,
            "input_tokens": int(attention_mask.sum().item()),
            "output_tokens": sum(len(row) for row in generated),
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
        "outputs": [
            {
                "receiver_token_ids": ids,
                "text": target_tokenizer.decode(ids, skip_special_tokens=True),
            }
            for ids in generated
        ],
    }


def build_smoke_report(
    wrapper: TrainingFreeTransportModel,
    source_tokenizer: Any,
    target_tokenizer: Any,
    *,
    prompt: str,
    generation: Mapping[str, Any],
    config: TransportConfig,
    code_version: str,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt:
        raise SmokeError("prompt must be a nonempty string")
    generation = _validate_generation(generation)

    receiver_only = _receiver_only_report(wrapper, target_tokenizer, prompt, generation)

    encoded = source_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = torch.as_tensor(
        encoded.get("attention_mask", torch.ones_like(input_ids)), dtype=torch.long
    )
    if attention_mask.ndim == 1:
        attention_mask = attention_mask.unsqueeze(0)
    device = _source_device(wrapper.source_model)
    output = wrapper.generate(
        input_ids.to(device),
        source_attention_mask=attention_mask.to(device),
        return_transport_output=True,
        **generation,
    )
    if not isinstance(output, TransportGenerationOutput):
        raise SmokeError("wrapper did not return structured transport diagnostics")
    token_rows = output.sequences.detach().cpu().tolist()
    decoded = [
        target_tokenizer.decode(row, skip_special_tokens=True) for row in token_rows
    ]
    artifact = wrapper.artifact
    fingerprint_payload = {
        "prompt": prompt,
        "config": config.to_dict(),
        "artifact_input_fingerprint": artifact.metadata.get("input_fingerprint"),
        "code_version": code_version,
        "runtime": dict(runtime or {}),
    }
    input_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 2,
        "input_fingerprint": input_fingerprint,
        "code_version": code_version,
        "prompt": prompt,
        "config": config.to_dict(),
        "runtime": dict(runtime or {}),
        "receiver_only": receiver_only,
        "artifact": {
            "shape": list(artifact.shape),
            "nnz": int(artifact.data.size),
            "metadata": artifact.metadata,
        },
        "shapes": {
            "source_input_ids": list(input_ids.shape),
            "virtual_prompt": list(output.virtual_prompt_shape),
            "receiver_output_ids": list(output.sequences.shape),
        },
        "transport_quality": {
            "retained_mass": _tensor_summary(
                output.stats.retained_mass, attention_mask
            ),
            "dropped_top_m_mass": _tensor_summary(
                output.stats.dropped_top_m_mass, attention_mask
            ),
            "active_support_mass": _tensor_summary(
                output.stats.active_support_mass, attention_mask
            ),
            "top_m": output.stats.top_m,
        },
        "metrics": output.metrics.to_dict(),
        "outputs": [
            {"receiver_token_ids": ids, "text": text}
            for ids, text in zip(token_rows, decoded)
        ],
    }


def save_smoke_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise SmokeError(f"refusing to overwrite smoke output: {path}")
    partial.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--code-version")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-locked-runtime", action="store_true")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=20.0)
    parser.add_argument(
        "--runtime-profile",
        choices=sorted(RUNTIME_PROFILES),
        default="project-cu124",
    )
    return parser.parse_args(argv)


def _load_runtime(config: TransportConfig, artifact_path: Path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    source_tokenizer = AutoTokenizer.from_pretrained(
        config.source.name,
        revision=config.source.tokenizer_revision,
        use_fast=True,
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        config.target.name,
        revision=config.target.tokenizer_revision,
        use_fast=True,
    )
    artifact = load_transport_artifact(
        artifact_path,
        source_fingerprint=tokenizer_fingerprint(source_tokenizer),
        target_fingerprint=tokenizer_fingerprint(target_tokenizer),
    )
    source_model = AutoModelForCausalLM.from_pretrained(
        config.source.name,
        revision=config.source.revision,
        torch_dtype=dtype[config.source.dtype],
        device_map=config.source.device_map,
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        config.target.name,
        revision=config.target.revision,
        torch_dtype=dtype[config.target.dtype],
        device_map=config.target.device_map,
    )
    wrapper = TrainingFreeTransportModel(
        source_model,
        target_model,
        artifact,
        tau=config.transport.tau,
        causal_shift=config.transport.causal_shift,
        source_top_m=config.transport.source_top_m,
    )
    return wrapper, source_tokenizer, target_tokenizer


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = TransportConfig.from_dict(
        yaml.safe_load(args.config.read_text(encoding="utf-8"))
    )
    prompt = (
        args.prompt
        if args.prompt is not None
        else args.prompt_file.read_text(encoding="utf-8")
    )
    validate_runtime_requirements(
        args.artifact,
        args.output,
        require_cuda=args.require_cuda,
        require_locked_runtime=args.require_locked_runtime,
        min_gpu_memory_gib=args.min_gpu_memory_gib,
        runtime_profile=args.runtime_profile,
    )
    torch.manual_seed(config.seed)
    wrapper, source_tokenizer, target_tokenizer = _load_runtime(config, args.artifact)
    report = build_smoke_report(
        wrapper,
        source_tokenizer,
        target_tokenizer,
        prompt=prompt,
        generation=config.generation,
        config=config,
        code_version=args.code_version or _git_version(),
        runtime=runtime_metadata(args.runtime_profile),
    )
    save_smoke_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
