"""Training-free transport adapter for the unified evaluator."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from time import perf_counter
from typing import Any, Mapping

import torch
import yaml

from rosetta.transport.config import CollaborationSpec, TransportConfig
from rosetta.transport.corpus import file_sha256
from rosetta.transport.approximations import precompute_source_values
from rosetta.transport.evaluation import (
    EvaluationSample,
    GenerationResult,
)
from rosetta.transport.wrapper import TransportGenerationOutput
from rosetta.transport.orf import build_orf_transport_state
from rosetta.transport.wrapper import TrainingFreeTransportModel
from script.transport.smoke_stt import (
    _git_version,
    _load_runtime,
    runtime_metadata,
    validate_runtime_requirements,
)


class TrainingFreeTransportEvaluationAdapter:
    method = "training_free_transport"

    def __init__(
        self,
        wrapper: Any,
        source_tokenizer: Any,
        target_tokenizer: Any,
        generation: Mapping[str, Any],
        provenance: Mapping[str, Any] | None = None,
        sender_generation: Mapping[str, Any] | None = None,
        collaboration: CollaborationSpec | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.generation = dict(generation)
        self.sender_generation = dict(sender_generation or {})
        self.collaboration = collaboration or CollaborationSpec()
        self.collaboration.validate()
        self.provenance = dict(provenance or {})

    @staticmethod
    def _role_messages(
        canonical_messages: list[Mapping[str, str]], system_prompt: str
    ) -> list[dict[str, str]]:
        messages = [dict(message) for message in canonical_messages]
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = system_prompt + "\n\n" + messages[0]["content"]
        else:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    @staticmethod
    def _encode(tokenizer: Any, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        attention_mask = torch.as_tensor(
            encoded.get("attention_mask", torch.ones_like(input_ids)),
            dtype=torch.long,
        )
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return input_ids, attention_mask

    def generate_one(self, sample: EvaluationSample) -> GenerationResult:
        canonical_messages = [dict(message) for message in sample.canonical_messages]
        sender_messages = self._role_messages(
            canonical_messages, self.collaboration.sender_system_prompt
        )
        receiver_messages = self._role_messages(
            canonical_messages, self.collaboration.receiver_system_prompt
        )
        source_text = self.source_tokenizer.apply_chat_template(
            sender_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.collaboration.sender_enable_thinking,
        )
        receiver_text = self.target_tokenizer.apply_chat_template(
            receiver_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.collaboration.receiver_enable_thinking,
        )
        source_ids, source_mask = self._encode(self.source_tokenizer, source_text)
        receiver_ids, receiver_mask = self._encode(self.target_tokenizer, receiver_text)
        device = self.wrapper.source_model.get_input_embeddings().weight.device
        source_ids = source_ids.to(device)
        source_mask = source_mask.to(device)
        if not self.sender_generation:
            raise ValueError("sender_generation must be configured")
        if self.sender_generation.get("do_sample") is not False:
            raise ValueError("sender generation must be greedy")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sender_start = perf_counter()
        with torch.no_grad():
            sender_context_ids = self.wrapper.source_model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                **self.sender_generation,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sender_generation_seconds = perf_counter() - sender_start
        if not isinstance(sender_context_ids, torch.Tensor):
            sender_context_ids = getattr(sender_context_ids, "sequences", None)
        if (
            not isinstance(sender_context_ids, torch.Tensor)
            or sender_context_ids.ndim != 2
            or sender_context_ids.shape[0] != source_ids.shape[0]
            or sender_context_ids.shape[1] <= source_ids.shape[1]
            or not torch.equal(sender_context_ids[:, : source_ids.shape[1]], source_ids)
        ):
            raise ValueError(
                "sender generate must return the prompt prefix plus at least one think token"
            )
        sender_think_ids = sender_context_ids[:, source_ids.shape[1] :]
        sender_context_mask = torch.cat(
            (
                source_mask,
                torch.ones_like(sender_think_ids, dtype=source_mask.dtype),
            ),
            dim=1,
        )
        output = self.wrapper.generate(
            sender_context_ids,
            source_attention_mask=sender_context_mask,
            receiver_input_ids=receiver_ids,
            receiver_attention_mask=receiver_mask,
            return_transport_output=True,
            **self.generation,
        )
        if not isinstance(output, TransportGenerationOutput):
            raise TypeError("transport adapter requires structured generation output")
        token_ids = output.sequences[0].detach().cpu().tolist()
        sender_think_token_ids = sender_think_ids[0].detach().cpu().tolist()
        sender_think_text = self.source_tokenizer.decode(
            sender_think_token_ids, skip_special_tokens=True
        )
        stats = output.stats
        metrics = output.metrics.to_dict()
        metrics["planner_generation_seconds"] = sender_generation_seconds
        metrics["sender_context_forward_seconds"] = metrics["source_seconds"]
        metrics["total_seconds"] += sender_generation_seconds
        metrics["sender_prompt_tokens"] = int(source_mask.sum().item())
        metrics["sender_think_tokens"] = len(sender_think_token_ids)
        metrics["sender_context_tokens"] = int(sender_context_mask.sum().item())
        metrics["aligned_sender_tokens"] = int(sender_context_mask.sum().item())
        diagnostics = {
            "virtual_prompt_shape": list(output.virtual_prompt_shape),
            "aligned_sender_shape": list(output.aligned_sender_shape),
            "receiver_prompt_shape": list(output.receiver_prompt_shape),
            "approximation_mode": getattr(self.wrapper, "approximation_mode", "exact"),
            "transport_stats_available": stats is not None,
            "retained_mass_mean": (
                None
                if stats is None
                else float(stats.retained_mass.float().mean().item())
            ),
            "dropped_top_m_mass_mean": (
                None
                if stats is None
                else float(stats.dropped_top_m_mass.float().mean().item())
            ),
            "active_support_mass_mean": (
                None
                if stats is None
                else float(stats.active_support_mass.float().mean().item())
            ),
            "source_top_m": None if stats is None else stats.top_m,
            "source_prompt_rendered": False,
            "sender_role": self.collaboration.sender_role,
            "receiver_role": self.collaboration.receiver_role,
            "sender_enable_thinking": self.collaboration.sender_enable_thinking,
            "receiver_enable_thinking": self.collaboration.receiver_enable_thinking,
            "sender_rendered_prompt": source_text,
            "sender_think_text": sender_think_text,
            "sender_think_token_ids": sender_think_token_ids,
            "receiver_rendered_prompt": receiver_text,
            "prefix_order": [
                "aligned_sender_prompt",
                "aligned_sender_think",
                "receiver_native_prompt",
            ],
            "provenance": self.provenance,
        }
        return GenerationResult(
            text=self.target_tokenizer.decode(token_ids, skip_special_tokens=True),
            token_ids=token_ids,
            metrics=metrics,
            diagnostics=diagnostics,
        )


def _configure_approximation(
    wrapper: TrainingFreeTransportModel, approximation: Mapping[str, Any]
) -> TrainingFreeTransportModel:
    allowed = {"mode", "source_top_m", "feature_count", "seed", "source_chunk_size"}
    unknown = set(approximation) - allowed
    if unknown:
        raise ValueError(f"unknown approximation fields: {sorted(unknown)}")
    mode = str(approximation.get("mode", "exact"))
    common = {
        "tau": wrapper.tau,
        "causal_shift": wrapper.causal_shift,
        "receiver_start_token_id": wrapper.receiver_start_token_id,
        "source_vocab_size": wrapper.source_vocab_size,
        "approximation_mode": mode,
        "source_fingerprint": wrapper.source_fingerprint,
        "target_fingerprint": wrapper.target_fingerprint,
    }
    if mode == "top_m":
        common["source_top_m"] = approximation.get("source_top_m")
    elif "source_top_m" in approximation:
        raise ValueError("source_top_m is only valid for top_m approximation")

    receiver_weight = wrapper.receiver_model.get_input_embeddings().weight
    if mode == "precomputed":
        common["precomputed_source_values"] = precompute_source_values(
            wrapper.artifact, receiver_weight
        )
    elif mode == "orf":
        if "feature_count" not in approximation:
            raise ValueError("orf approximation requires feature_count")
        output_layer = wrapper.source_model.get_output_embeddings()
        if output_layer is None or getattr(output_layer, "weight", None) is None:
            raise ValueError("orf approximation requires source output embeddings")
        output_weight = output_layer.weight.detach().cpu()
        output_bias = getattr(output_layer, "bias", None)
        if output_bias is not None:
            output_bias = output_bias.detach().cpu()
        state = build_orf_transport_state(
            output_weight,
            output_bias,
            wrapper.artifact,
            receiver_weight.detach().cpu(),
            feature_count=approximation["feature_count"],
            tau=wrapper.tau,
            seed=approximation.get("seed", 42),
            source_chunk_size=approximation.get("source_chunk_size", 1_024),
            source_vocab_size=wrapper.source_vocab_size,
        )
        common["orf_state"] = replace(
            state,
            omega=state.omega.to(receiver_weight.device),
            numerator=state.numerator.to(receiver_weight.device),
            denominator=state.denominator.to(receiver_weight.device),
        )
    elif any(
        field in approximation
        for field in ("feature_count", "seed", "source_chunk_size")
    ):
        raise ValueError("ORF parameters are only valid for orf approximation")
    return TrainingFreeTransportModel(
        wrapper.source_model,
        wrapper.receiver_model,
        wrapper.artifact,
        **common,
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
    sender_generation = dict(config.sender_generation)
    sender_generation.update(model_config.get("sender_generation_config", {}))
    wrapper, source_tokenizer, target_tokenizer = _load_runtime(
        config,
        artifact_path,
        source_device_map=model_config.get("source_device_map"),
        target_device_map=model_config.get("target_device_map"),
    )
    approximation = model_config.get("approximation")
    if approximation is not None:
        if not isinstance(approximation, Mapping):
            raise ValueError("approximation must be a mapping")
        wrapper = _configure_approximation(wrapper, approximation)
    runtime_profile = str(model_config.get("runtime_profile", "project-cu124"))
    provenance = {
        "code_version": _git_version(),
        "runtime": runtime_metadata(runtime_profile),
        "transport_config_path": str(config_path),
        "transport_config": config.to_dict(),
        "source_device_map_override": model_config.get("source_device_map"),
        "target_device_map_override": model_config.get("target_device_map"),
        "artifact_path": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "artifact_shape": list(wrapper.artifact.shape),
        "artifact_nnz": int(wrapper.artifact.data.size),
        "artifact_metadata": dict(wrapper.artifact.metadata),
        "approximation": {
            "mode": getattr(wrapper, "approximation_mode", "exact"),
            **({} if approximation is None else dict(approximation)),
        },
    }
    return TrainingFreeTransportEvaluationAdapter(
        wrapper,
        source_tokenizer,
        target_tokenizer,
        generation,
        provenance=provenance,
        sender_generation=sender_generation,
        collaboration=config.collaboration,
    )
