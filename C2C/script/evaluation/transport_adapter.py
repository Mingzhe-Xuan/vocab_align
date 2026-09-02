"""Training-free transport adapter for the unified evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from rosetta.transport.config import TransportConfig
from rosetta.transport.evaluation import (
    EvaluationSample,
    GenerationResult,
)
from rosetta.transport.wrapper import TransportGenerationOutput
from script.transport.smoke_stt import _load_runtime, validate_runtime_requirements


class TrainingFreeTransportEvaluationAdapter:
    method = "training_free_transport"

    def __init__(
        self,
        wrapper: Any,
        source_tokenizer: Any,
        target_tokenizer: Any,
        generation: Mapping[str, Any],
    ) -> None:
        self.wrapper = wrapper
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.generation = dict(generation)

    def generate_one(self, sample: EvaluationSample) -> GenerationResult:
        source_text = self.source_tokenizer.apply_chat_template(
            list(sample.canonical_messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = self.source_tokenizer(source_text, return_tensors="pt")
        source_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        if source_ids.ndim == 1:
            source_ids = source_ids.unsqueeze(0)
        attention_mask = torch.as_tensor(
            encoded.get("attention_mask", torch.ones_like(source_ids)),
            dtype=torch.long,
        )
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        device = self.wrapper.source_model.get_input_embeddings().weight.device
        output = self.wrapper.generate(
            source_ids.to(device),
            source_attention_mask=attention_mask.to(device),
            return_transport_output=True,
            **self.generation,
        )
        if not isinstance(output, TransportGenerationOutput):
            raise TypeError("transport adapter requires structured generation output")
        token_ids = output.sequences[0].detach().cpu().tolist()
        stats = output.stats
        diagnostics = {
            "virtual_prompt_shape": list(output.virtual_prompt_shape),
            "retained_mass_mean": float(stats.retained_mass.float().mean().item()),
            "dropped_top_m_mass_mean": float(
                stats.dropped_top_m_mass.float().mean().item()
            ),
            "active_support_mass_mean": float(
                stats.active_support_mass.float().mean().item()
            ),
            "source_top_m": stats.top_m,
            "source_rendered_prompt": source_text,
        }
        return GenerationResult(
            text=self.target_tokenizer.decode(token_ids, skip_special_tokens=True),
            token_ids=token_ids,
            metrics=output.metrics.to_dict(),
            diagnostics=diagnostics,
        )


def create_training_free_transport_adapter(
    model_config: Mapping[str, Any],
) -> TrainingFreeTransportEvaluationAdapter:
    if model_config.get("model_name") != "training_free_transport":
        raise ValueError(
            "transport adapter requires model_name=training_free_transport"
        )
    config_path = Path(str(model_config["transport_config"]))
    artifact_path = Path(str(model_config["artifact"]))
    validate_runtime_requirements(
        artifact_path,
        artifact_path,
        require_cuda=bool(model_config.get("require_cuda", True)),
        require_locked_runtime=bool(model_config.get("require_locked_runtime", True)),
        min_gpu_memory_gib=float(model_config.get("min_gpu_memory_gib", 20.0)),
        runtime_profile=str(model_config.get("runtime_profile", "project-cu124")),
        allow_existing_output=True,
    )
    config = TransportConfig.from_dict(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    generation = dict(config.generation)
    generation.update(model_config.get("generation_config", {}))
    wrapper, source_tokenizer, target_tokenizer = _load_runtime(config, artifact_path)
    return TrainingFreeTransportEvaluationAdapter(
        wrapper, source_tokenizer, target_tokenizer, generation
    )
