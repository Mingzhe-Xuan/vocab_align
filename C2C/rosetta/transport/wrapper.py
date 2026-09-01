"""Training-free source-logit transport and receiver-native generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Tokenizer/audit-only environments do not need torch.
    torch = None  # type: ignore[assignment]

from .artifact import TransportArtifact
from .soft_transport import SoftTransportStats, transport_embeddings


_ModuleBase = torch.nn.Module if torch is not None else object


class TransportModelError(ValueError):
    """Raised when the source/receiver transport protocol is invalid."""


@dataclass(frozen=True)
class VirtualPrompt:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    stats: SoftTransportStats


@dataclass(frozen=True)
class TransportPrefill:
    virtual_prompt: VirtualPrompt
    receiver_output: Any


def _require_torch() -> None:
    if torch is None:
        raise TransportModelError("TrainingFreeTransportModel requires PyTorch")


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return (attention_mask.long().cumsum(dim=-1) - 1).clamp_min(0)


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
    ) -> None:
        _require_torch()
        super().__init__()
        artifact.validate()
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
        self.source_model = source_model
        self.receiver_model = receiver_model
        self.artifact = artifact
        self.tau = float(tau)
        self.causal_shift = causal_shift
        self.source_top_m = source_top_m
        self.receiver_start_token_id = receiver_start_token_id

    def build_virtual_prompt(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor | None = None,
    ) -> VirtualPrompt:
        if source_attention_mask is None:
            source_attention_mask = torch.ones_like(source_input_ids, dtype=torch.long)
        _validate_inputs(source_input_ids, source_attention_mask)
        position_ids = _position_ids(source_attention_mask)
        with torch.no_grad():
            source_output = self.source_model(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
            )
        logits = getattr(source_output, "logits", None)
        if logits is None or logits.shape[:2] != source_input_ids.shape:
            raise TransportModelError(
                "source output logits must match batch and sequence"
            )
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise TransportModelError("source logits must be finite floating values")

        receiver_embeddings = self.receiver_model.get_input_embeddings()
        receiver_weight = receiver_embeddings.weight
        transport_logits = logits.to(device=receiver_weight.device, dtype=torch.float32)
        source_attention_mask = source_attention_mask.to(receiver_weight.device)
        position_ids = position_ids.to(receiver_weight.device)
        transported, _, stats = transport_embeddings(
            transport_logits,
            self.artifact,
            receiver_weight,
            tau=self.tau,
            top_m=self.source_top_m,
        )
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
        return VirtualPrompt(embeddings, source_attention_mask, position_ids, stats)

    def prefill(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor | None = None,
    ) -> TransportPrefill:
        virtual_prompt = self.build_virtual_prompt(
            source_input_ids, source_attention_mask
        )
        with torch.no_grad():
            receiver_output = self.receiver_model(
                inputs_embeds=virtual_prompt.embeddings,
                attention_mask=virtual_prompt.attention_mask,
                position_ids=virtual_prompt.position_ids,
                use_cache=True,
                return_dict=True,
            )
        if getattr(receiver_output, "past_key_values", None) is None:
            raise TransportModelError("receiver prefill did not return a KV cache")
        logits = getattr(receiver_output, "logits", None)
        if logits is None or logits.shape[:2] != source_input_ids.shape:
            raise TransportModelError(
                "receiver prefill logits must match virtual prompt"
            )
        return TransportPrefill(virtual_prompt, receiver_output)

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
        **kwargs: Any,
    ) -> torch.Tensor:
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise TransportModelError("max_new_tokens must be a nonnegative integer")
        if not transport:
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
            prefill = self.prefill(source_input_ids, source_attention_mask)
            mask = prefill.virtual_prompt.attention_mask
            output = prefill.receiver_output
            batch = source_input_ids.shape[0]
            sequences = torch.empty((batch, 0), dtype=torch.long, device=mask.device)
            if max_new_tokens == 0:
                return sequences
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

            for step in range(max_new_tokens):
                was_active = ~finished
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
            return sequences
