"""Training-free source-logit transport and receiver-native generation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Tokenizer/audit-only environments do not need torch.
    torch = None  # type: ignore[assignment]

from .artifact import TransportArtifact
from .approximations import (
    ApproximationError,
    hard_transport_embeddings,
    precomputed_transport_embeddings,
)
from .metrics import TransportMetrics
from .orf import OrfError, OrfTransportState, apply_orf_transport
from .soft_transport import SoftTransportStats, transport_embeddings


_ModuleBase = torch.nn.Module if torch is not None else object
_TRANSPORT_QUERY_CHUNK_SIZE = 32


class TransportModelError(ValueError):
    """Raised when the source/receiver transport protocol is invalid."""


@dataclass(frozen=True)
class VirtualPrompt:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    stats: SoftTransportStats | None


@dataclass(frozen=True)
class TransportPrefill:
    virtual_prompt: VirtualPrompt
    receiver_output: Any


@dataclass(frozen=True)
class TransportGenerationOutput:
    sequences: torch.Tensor
    virtual_prompt_shape: Tuple[int, ...]
    stats: SoftTransportStats | None
    metrics: TransportMetrics


def _require_torch() -> None:
    if torch is None:
        raise TransportModelError("TrainingFreeTransportModel requires PyTorch")


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    if input_ids.ndim != 2:
        raise TransportModelError("input_ids must have shape [batch, sequence]")
    if attention_mask.shape != input_ids.shape:
        raise TransportModelError("attention_mask must match input_ids")
    if torch.any((attention_mask != 0) & (attention_mask != 1)):
        raise TransportModelError("attention_mask must be binary")
    for row in attention_mask.bool():
        active = torch.nonzero(row, as_tuple=False).flatten()
        if active.numel() == 0:
            raise TransportModelError("every sequence must contain an active token")
        expected = torch.arange(active[0], active[-1] + 1, device=active.device)
        if not torch.equal(active, expected):
            raise TransportModelError("attention_mask active tokens must be contiguous")


def _last_active_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return (
        positions.expand_as(attention_mask)
        .masked_fill(~attention_mask.bool(), -1)
        .max(-1)
        .values
    )


def _sample_next_token(
    logits: torch.Tensor,
    *,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise TransportModelError("receiver logits must be finite and rank two")
    if not do_sample:
        return logits.argmax(dim=-1)
    if not np.isfinite(temperature) or temperature <= 0:
        raise TransportModelError("generation temperature must be finite and positive")
    if not 0 < top_p <= 1:
        raise TransportModelError("top_p must be in (0, 1]")
    if (
        isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k == 0
        or top_k < -1
    ):
        raise TransportModelError("top_k must be -1 or a positive integer")
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_k != -1:
        values, indices = torch.topk(
            probabilities, min(top_k, probabilities.shape[-1]), dim=-1
        )
        filtered = torch.zeros_like(probabilities).scatter(-1, indices, values)
        probabilities = filtered / filtered.sum(dim=-1, keepdim=True)
    if top_p < 1:
        values, indices = probabilities.sort(dim=-1, descending=True)
        cumulative = values.cumsum(dim=-1)
        keep = cumulative - values < top_p
        values = values * keep
        values = values / values.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(values, 1)
        return indices.gather(-1, sampled).squeeze(-1)
    return torch.multinomial(probabilities, 1).squeeze(-1)


class TrainingFreeTransportModel(_ModuleBase):
    """Run source prefill once, transport logits, then decode only with receiver B."""

    def __init__(
        self,
        source_model: Any,
        receiver_model: Any,
        artifact: TransportArtifact,
        *,
        tau: float,
        causal_shift: bool = True,
        source_top_m: int | None = None,
        receiver_start_token_id: int | None = None,
        source_vocab_size: int | None = None,
        approximation_mode: str | None = None,
        precomputed_source_values: torch.Tensor | None = None,
        orf_state: OrfTransportState | None = None,
        source_fingerprint: str | None = None,
        target_fingerprint: str | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        artifact.validate()
        if (source_fingerprint is None) != (target_fingerprint is None):
            raise TransportModelError(
                "source and target fingerprints must be provided together"
            )
        if source_fingerprint is not None:
            if not source_fingerprint or not target_fingerprint:
                raise TransportModelError("tokenizer fingerprints must be nonempty")
            if artifact.metadata["source_fingerprint"] != source_fingerprint:
                raise TransportModelError("source tokenizer fingerprint mismatch")
            if artifact.metadata["target_fingerprint"] != target_fingerprint:
                raise TransportModelError("target tokenizer fingerprint mismatch")
        if not np.isfinite(tau) or tau <= 0:
            raise TransportModelError("transport tau must be finite and positive")
        if not isinstance(causal_shift, bool):
            raise TransportModelError("causal_shift must be boolean")
        if source_top_m is not None and (
            isinstance(source_top_m, bool)
            or not isinstance(source_top_m, int)
            or source_top_m <= 0
        ):
            raise TransportModelError("source_top_m must be null or a positive integer")
        if source_vocab_size is not None and (
            isinstance(source_vocab_size, bool)
            or not isinstance(source_vocab_size, int)
            or source_vocab_size <= 0
        ):
            raise TransportModelError(
                "source_vocab_size must be null or a positive integer"
            )
        if approximation_mode is None:
            approximation_mode = "top_m" if source_top_m is not None else "exact"
        if approximation_mode not in {"exact", "hard", "top_m", "precomputed", "orf"}:
            raise TransportModelError("unsupported approximation mode")
        if approximation_mode == "top_m" and source_top_m is None:
            raise TransportModelError("top_m mode requires source_top_m")
        if approximation_mode != "top_m" and source_top_m is not None:
            raise TransportModelError(
                "source_top_m is only valid for the top_m approximation mode"
            )
        if approximation_mode == "precomputed":
            if precomputed_source_values is None:
                raise TransportModelError(
                    "precomputed mode requires precomputed_source_values"
                )
        elif precomputed_source_values is not None:
            raise TransportModelError(
                "precomputed_source_values is only valid in precomputed mode"
            )
        if approximation_mode == "orf":
            if orf_state is None:
                raise TransportModelError("orf mode requires orf_state")
        elif orf_state is not None:
            raise TransportModelError("orf_state is only valid in orf mode")
        if receiver_start_token_id is None:
            config = getattr(receiver_model, "config", None)
            receiver_start_token_id = getattr(config, "bos_token_id", None)
        if causal_shift and receiver_start_token_id is None:
            raise TransportModelError("causal shift requires a receiver start token ID")
        if receiver_start_token_id is not None and (
            isinstance(receiver_start_token_id, bool)
            or not isinstance(receiver_start_token_id, int)
            or receiver_start_token_id < 0
        ):
            raise TransportModelError("receiver start token ID must be nonnegative")
        receiver_vocab = receiver_model.get_input_embeddings().weight.shape[0]
        if (
            receiver_start_token_id is not None
            and receiver_start_token_id >= receiver_vocab
        ):
            raise TransportModelError(
                "receiver start token ID exceeds receiver vocabulary"
            )
        receiver_weight = receiver_model.get_input_embeddings().weight
        if precomputed_source_values is not None:
            if (
                precomputed_source_values.ndim != 2
                or precomputed_source_values.shape
                != (artifact.shape[1], receiver_weight.shape[1])
            ):
                raise TransportModelError(
                    "precomputed_source_values shape must match source rows and receiver dimension"
                )
            if precomputed_source_values.device != receiver_weight.device:
                raise TransportModelError(
                    "precomputed_source_values must share the receiver embedding device"
                )
            if (
                not precomputed_source_values.is_floating_point()
                or not torch.isfinite(precomputed_source_values).all()
            ):
                raise TransportModelError(
                    "precomputed_source_values must be finite floating values"
                )
        if orf_state is not None:
            state_tensors = (
                orf_state.omega,
                orf_state.numerator,
                orf_state.denominator,
            )
            if any(tensor.device != receiver_weight.device for tensor in state_tensors):
                raise TransportModelError(
                    "ORF state must share the receiver embedding device"
                )
            if orf_state.numerator.shape[0] != receiver_weight.shape[1]:
                raise TransportModelError("ORF receiver dimension mismatch")
            if orf_state.target_vocab_size != receiver_vocab:
                raise TransportModelError("ORF target vocabulary mismatch")
            expected_source_vocab = source_vocab_size or artifact.shape[1]
            if orf_state.source_vocab_size != expected_source_vocab:
                raise TransportModelError("ORF source vocabulary mismatch")
            if orf_state.tau != float(tau):
                raise TransportModelError("ORF state tau mismatch")
            if orf_state.source_fingerprint != str(
                artifact.metadata["source_fingerprint"]
            ) or orf_state.target_fingerprint != str(
                artifact.metadata["target_fingerprint"]
            ):
                raise TransportModelError("ORF state fingerprint mismatch")
        self.source_model = source_model
        self.receiver_model = receiver_model
        self.artifact = artifact
        self.tau = float(tau)
        self.causal_shift = causal_shift
        self.source_top_m = source_top_m
        self.source_vocab_size = source_vocab_size
        self.receiver_start_token_id = receiver_start_token_id
        self.approximation_mode = approximation_mode
        self.precomputed_source_values = precomputed_source_values
        self.orf_state = orf_state
        self.source_fingerprint = source_fingerprint
        self.target_fingerprint = target_fingerprint

    def build_virtual_prompt(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor | None = None,
    ) -> VirtualPrompt:
        virtual_prompt, _, _ = self._build_virtual_prompt_timed(
            source_input_ids, source_attention_mask
        )
        return virtual_prompt

    def _build_virtual_prompt_timed(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor | None,
    ) -> tuple[VirtualPrompt, float, float]:
        if source_attention_mask is None:
            source_attention_mask = torch.ones_like(source_input_ids, dtype=torch.long)
        _validate_inputs(source_input_ids, source_attention_mask)
        position_ids = _position_ids(source_attention_mask)
        _synchronize(source_input_ids.device)
        source_start = perf_counter()
        with torch.no_grad():
            source_callable = self.source_model
            if self.approximation_mode == "orf":
                source_callable = getattr(self.source_model, "model", None)
                if source_callable is None:
                    raise TransportModelError(
                        "orf mode requires a source backbone exposed as source_model.model"
                    )
            source_output = source_callable(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        _synchronize(source_input_ids.device)
        source_seconds = perf_counter() - source_start

        receiver_embeddings = self.receiver_model.get_input_embeddings()
        receiver_weight = receiver_embeddings.weight
        _synchronize(receiver_weight.device)
        transport_start = perf_counter()
        source_attention_mask = source_attention_mask.to(receiver_weight.device)
        position_ids = position_ids.to(receiver_weight.device)
        stats = None
        try:
            if self.approximation_mode == "orf":
                hidden = getattr(source_output, "last_hidden_state", None)
                if hidden is None or hidden.shape[:2] != source_input_ids.shape:
                    raise TransportModelError(
                        "source backbone hidden state must match batch and sequence"
                    )
                if not hidden.is_floating_point() or not torch.isfinite(hidden).all():
                    raise TransportModelError(
                        "source backbone hidden state must contain finite floating values"
                    )
                transported = apply_orf_transport(
                    hidden.to(device=receiver_weight.device, dtype=torch.float32),
                    self.orf_state,
                    source_fingerprint=str(
                        self.artifact.metadata["source_fingerprint"]
                    ),
                    target_fingerprint=str(
                        self.artifact.metadata["target_fingerprint"]
                    ),
                )
            else:
                logits = getattr(source_output, "logits", None)
                if logits is None or logits.shape[:2] != source_input_ids.shape:
                    raise TransportModelError(
                        "source output logits must match batch and sequence"
                    )
                if not logits.is_floating_point() or not torch.isfinite(logits).all():
                    raise TransportModelError(
                        "source logits must be finite floating values"
                    )
                if self.approximation_mode in {"exact", "top_m"}:
                    transported_chunks = []
                    stats_chunks = []
                    for start in range(0, logits.shape[1], _TRANSPORT_QUERY_CHUNK_SIZE):
                        stop = min(start + _TRANSPORT_QUERY_CHUNK_SIZE, logits.shape[1])
                        chunk, _, chunk_stats = transport_embeddings(
                            logits[:, start:stop].to(
                                device=receiver_weight.device, dtype=torch.float32
                            ),
                            self.artifact,
                            receiver_weight,
                            tau=self.tau,
                            top_m=self.source_top_m,
                            source_vocab_size=self.source_vocab_size,
                        )
                        transported_chunks.append(chunk)
                        stats_chunks.append(chunk_stats)
                    transported = torch.cat(transported_chunks, dim=1)
                    stats = SoftTransportStats(
                        retained_mass=torch.cat(
                            [item.retained_mass for item in stats_chunks], dim=1
                        ),
                        dropped_top_m_mass=torch.cat(
                            [item.dropped_top_m_mass for item in stats_chunks], dim=1
                        ),
                        active_support_mass=torch.cat(
                            [item.active_support_mass for item in stats_chunks], dim=1
                        ),
                        top_m=stats_chunks[0].top_m,
                    )
                elif self.approximation_mode == "hard":
                    transport_logits = logits.to(
                        device=receiver_weight.device, dtype=torch.float32
                    )
                    transported, _, stats = hard_transport_embeddings(
                        transport_logits,
                        self.artifact,
                        receiver_weight,
                        tau=self.tau,
                        source_vocab_size=self.source_vocab_size,
                    )
                else:
                    transport_logits = logits.to(
                        device=receiver_weight.device, dtype=torch.float32
                    )
                    transported, stats = precomputed_transport_embeddings(
                        transport_logits,
                        self.artifact,
                        self.precomputed_source_values,
                        tau=self.tau,
                        source_vocab_size=self.source_vocab_size,
                    )
        except (ApproximationError, OrfError) as error:
            raise TransportModelError(str(error)) from error
        transported = transported.to(dtype=receiver_weight.dtype)
        active = source_attention_mask.bool().unsqueeze(-1)
        if self.causal_shift:
            shifted = torch.zeros_like(transported)
            shifted[:, 1:] = transported[:, :-1]
            first_active = source_attention_mask.bool().long().argmax(dim=-1)
            start_ids = torch.full(
                (source_input_ids.shape[0],),
                int(self.receiver_start_token_id),
                dtype=torch.long,
                device=receiver_weight.device,
            )
            start = receiver_embeddings(start_ids).to(dtype=receiver_weight.dtype)
            shifted[
                torch.arange(shifted.shape[0], device=shifted.device), first_active
            ] = start
            embeddings = shifted.masked_fill(~active, 0)
        else:
            embeddings = transported.masked_fill(~active, 0)
        _synchronize(receiver_weight.device)
        transport_seconds = perf_counter() - transport_start
        return (
            VirtualPrompt(embeddings, source_attention_mask, position_ids, stats),
            source_seconds,
            transport_seconds,
        )

    def prefill(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor | None = None,
    ) -> TransportPrefill:
        prefill, _, _, _ = self._prefill_timed(source_input_ids, source_attention_mask)
        return prefill

    def _prefill_timed(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor | None,
    ) -> tuple[TransportPrefill, float, float, float]:
        virtual_prompt, source_seconds, transport_seconds = (
            self._build_virtual_prompt_timed(source_input_ids, source_attention_mask)
        )
        receiver_device = virtual_prompt.embeddings.device
        _synchronize(receiver_device)
        receiver_start = perf_counter()
        with torch.no_grad():
            receiver_output = self.receiver_model(
                inputs_embeds=virtual_prompt.embeddings,
                attention_mask=virtual_prompt.attention_mask,
                position_ids=virtual_prompt.position_ids,
                use_cache=True,
                return_dict=True,
            )
        _synchronize(receiver_device)
        receiver_seconds = perf_counter() - receiver_start
        if getattr(receiver_output, "past_key_values", None) is None:
            raise TransportModelError("receiver prefill did not return a KV cache")
        logits = getattr(receiver_output, "logits", None)
        if logits is None or logits.shape[:2] != source_input_ids.shape:
            raise TransportModelError(
                "receiver prefill logits must match virtual prompt"
            )
        return (
            TransportPrefill(virtual_prompt, receiver_output),
            source_seconds,
            transport_seconds,
            receiver_seconds,
        )

    def generate(
        self,
        source_input_ids: torch.Tensor | None = None,
        *,
        source_attention_mask: torch.Tensor | None = None,
        receiver_input_ids: torch.Tensor | None = None,
        receiver_attention_mask: torch.Tensor | None = None,
        transport: bool = True,
        max_new_tokens: int,
        eos_token_id: int | Tuple[int, ...] | None = None,
        pad_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        return_transport_output: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | TransportGenerationOutput:
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise TransportModelError("max_new_tokens must be a nonnegative integer")
        if not transport:
            if return_transport_output:
                raise TransportModelError(
                    "structured transport output is unavailable in receiver-only mode"
                )
            if receiver_input_ids is None:
                raise TransportModelError(
                    "receiver-only generation requires receiver_input_ids"
                )
            return self.receiver_model.generate(
                input_ids=receiver_input_ids,
                attention_mask=receiver_attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **kwargs,
            )
        if source_input_ids is None:
            raise TransportModelError("transport generation requires source_input_ids")
        with torch.no_grad():
            receiver_device = self.receiver_model.get_input_embeddings().weight.device
            if receiver_device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(receiver_device)
            prefill, source_seconds, transport_seconds, receiver_prefill_seconds = (
                self._prefill_timed(source_input_ids, source_attention_mask)
            )
            mask = prefill.virtual_prompt.attention_mask
            output = prefill.receiver_output
            batch = source_input_ids.shape[0]
            sequences = torch.empty((batch, 0), dtype=torch.long, device=mask.device)
            generated_active = 0
            decode_seconds = 0.0
            if max_new_tokens > 0:
                last = _last_active_indices(mask)
                logits_device = output.logits.device
                logits = output.logits[
                    torch.arange(batch, device=logits_device), last.to(logits_device)
                ]
                cache = output.past_key_values

            config = getattr(self.receiver_model, "generation_config", None)
            if config is None:
                config = getattr(self.receiver_model, "config", None)
            if eos_token_id is None and config is not None:
                eos_token_id = getattr(config, "eos_token_id", None)
            if pad_token_id is None and config is not None:
                pad_token_id = getattr(config, "pad_token_id", None)
            eos_ids = set()
            if eos_token_id is not None:
                values = (
                    (eos_token_id,) if isinstance(eos_token_id, int) else eos_token_id
                )
                eos_ids = {int(value) for value in values}
            if pad_token_id is None:
                pad_token_id = min(eos_ids) if eos_ids else 0
            finished = torch.zeros(batch, dtype=torch.bool, device=mask.device)
            running_mask = mask

            if max_new_tokens > 0:
                _synchronize(receiver_device)
                decode_start = perf_counter()
                for step in range(max_new_tokens):
                    was_active = ~finished
                    generated_active += int(was_active.sum().item())
                    next_token = _sample_next_token(
                        logits,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                    ).to(mask.device)
                    next_token = torch.where(
                        was_active,
                        next_token,
                        torch.full_like(next_token, pad_token_id),
                    )
                    sequences = torch.cat((sequences, next_token.unsqueeze(1)), dim=1)
                    if eos_ids:
                        is_eos = torch.zeros_like(finished)
                        for token_id in eos_ids:
                            is_eos |= next_token == token_id
                        finished |= was_active & is_eos
                    if step + 1 == max_new_tokens or torch.all(finished):
                        break
                    running_mask = torch.cat(
                        (running_mask, was_active.long().unsqueeze(1)), dim=1
                    )
                    decode_position = running_mask.sum(dim=-1, keepdim=True) - 1
                    output = self.receiver_model(
                        input_ids=next_token.unsqueeze(1),
                        attention_mask=running_mask,
                        position_ids=decode_position,
                        past_key_values=cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    cache = getattr(output, "past_key_values", None)
                    if cache is None:
                        raise TransportModelError(
                            "receiver decode did not return a KV cache"
                        )
                    logits = output.logits[:, -1, :]
                _synchronize(receiver_device)
                decode_seconds = perf_counter() - decode_start
            if not return_transport_output:
                return sequences
            peak_memory = (
                torch.cuda.max_memory_allocated(receiver_device)
                if receiver_device.type == "cuda"
                else None
            )
            total_seconds = (
                source_seconds
                + transport_seconds
                + receiver_prefill_seconds
                + decode_seconds
            )
            metrics = TransportMetrics(
                source_seconds=source_seconds,
                transport_seconds=transport_seconds,
                receiver_prefill_seconds=receiver_prefill_seconds,
                decode_seconds=decode_seconds,
                total_seconds=total_seconds,
                source_input_tokens=int(mask.sum().item()),
                virtual_tokens=int(mask.sum().item()),
                output_tokens=generated_active,
                peak_memory_bytes=peak_memory,
            )
            metrics.validate()
            return TransportGenerationOutput(
                sequences=sequences,
                virtual_prompt_shape=tuple(prefill.virtual_prompt.embeddings.shape),
                stats=prefill.virtual_prompt.stats,
                metrics=metrics,
            )
