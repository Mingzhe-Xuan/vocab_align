"""Run one reproducible STT prompt and atomically save diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
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


def build_smoke_report(
    wrapper: TrainingFreeTransportModel,
    source_tokenizer: Any,
    target_tokenizer: Any,
    *,
    prompt: str,
    generation: Mapping[str, Any],
    config: TransportConfig,
    code_version: str,
) -> dict[str, Any]:
    if not isinstance(prompt, str) or not prompt:
        raise SmokeError("prompt must be a nonempty string")
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
    if "max_new_tokens" not in generation:
        raise SmokeError("generation requires max_new_tokens")

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
        **dict(generation),
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
        "schema_version": 1,
        "input_fingerprint": input_fingerprint,
        "code_version": code_version,
        "prompt": prompt,
        "config": config.to_dict(),
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
    wrapper, source_tokenizer, target_tokenizer = _load_runtime(config, args.artifact)
    report = build_smoke_report(
        wrapper,
        source_tokenizer,
        target_tokenizer,
        prompt=prompt,
        generation=config.generation,
        config=config,
        code_version=args.code_version or _git_version(),
    )
    save_smoke_report(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
