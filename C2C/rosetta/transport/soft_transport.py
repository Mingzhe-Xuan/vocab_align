"""Exact and source-top-m soft-token transport into receiver embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Tokenizer/audit-only environments do not need torch.
    torch = None  # type: ignore[assignment]

from .artifact import TransportArtifact


class SoftTransportError(ValueError):
    """Raised when logits, artifact, or receiver embeddings are incompatible."""


@dataclass(frozen=True)
class SoftTransportStats:
    retained_mass: torch.Tensor
    dropped_top_m_mass: torch.Tensor
    active_support_mass: torch.Tensor
    top_m: int | None


def _require_torch() -> None:
    if torch is None:
        raise SoftTransportError("soft transport requires PyTorch")


def _full_source_support(artifact: TransportArtifact, vocab_size: int) -> bool:
    return len(artifact.source_token_ids) == vocab_size and np.array_equal(
        np.sort(artifact.source_token_ids), np.arange(vocab_size)
    )


def transport_probabilities(
    logits: torch.Tensor,
    artifact: TransportArtifact,
    *,
    tau: float,
    top_m: int | None = None,
    allow_partial_support: bool = False,
    source_vocab_size: int | None = None,
) -> tuple[torch.Tensor, SoftTransportStats]:
    """Compute compact receiver probabilities ``T softmax(logits/tau)``.

    Truncated or partial active-support probabilities are renormalized, and
    their retained mass is returned so approximation loss is never silent.
    """
    normalized, stats = source_probabilities(
        logits,
        artifact,
        tau=tau,
        top_m=top_m,
        allow_partial_support=allow_partial_support,
        source_vocab_size=source_vocab_size,
    )

    columns_np = np.repeat(
        np.arange(artifact.shape[1], dtype=np.int64), np.diff(artifact.indptr)
    )
    rows = torch.as_tensor(artifact.indices, dtype=torch.long, device=logits.device)
    columns = torch.as_tensor(columns_np, dtype=torch.long, device=logits.device)
    weights = torch.as_tensor(artifact.data, dtype=logits.dtype, device=logits.device)
    contributions = normalized.index_select(-1, columns) * weights
    target = torch.zeros(
        (*logits.shape[:-1], artifact.shape[0]),
        dtype=logits.dtype,
        device=logits.device,
    )
    target.index_add_(-1, rows, contributions)
    return target, stats


def source_probabilities(
    logits: torch.Tensor,
    artifact: TransportArtifact,
    *,
    tau: float,
    top_m: int | None = None,
    allow_partial_support: bool = False,
    source_vocab_size: int | None = None,
) -> tuple[torch.Tensor, SoftTransportStats]:
    """Return normalized probabilities on artifact compact source columns."""
    _require_torch()
    artifact.validate()
    if logits.ndim < 1 or not logits.is_floating_point():
        raise SoftTransportError("logits must be a floating tensor")
    if not np.isfinite(tau) or tau <= 0:
        raise SoftTransportError("tau must be finite and positive")
    vocab_size = logits.shape[-1]
    if np.any(artifact.source_token_ids >= vocab_size):
        raise SoftTransportError("artifact source token ID exceeds logits vocabulary")
    if source_vocab_size is not None:
        if (
            isinstance(source_vocab_size, bool)
            or not isinstance(source_vocab_size, int)
            or not 1 <= source_vocab_size <= vocab_size
        ):
            raise SoftTransportError(
                "source_vocab_size must be in [1, logits vocabulary]"
            )
        if not _full_source_support(artifact, source_vocab_size):
            raise SoftTransportError(
                "artifact must fully cover the explicit tokenizer vocabulary"
            )
        transport_vocab_size = source_vocab_size
    else:
        transport_vocab_size = vocab_size
    if (
        source_vocab_size is None
        and not _full_source_support(artifact, vocab_size)
        and not allow_partial_support
    ):
        raise SoftTransportError(
            "exact transport requires artifact coverage of the full source vocabulary"
        )
    if top_m is not None and (
        isinstance(top_m, bool)
        or not isinstance(top_m, int)
        or not 1 <= top_m <= transport_vocab_size
    ):
        raise SoftTransportError("top_m must be an integer in [1, source_vocab]")

    probabilities = torch.softmax(logits[..., :transport_vocab_size] / tau, dim=-1)
    if top_m is None or top_m == transport_vocab_size:
        truncated = probabilities
        dropped_top_m = torch.zeros_like(probabilities[..., 0])
    else:
        values, indices = torch.topk(probabilities, top_m, dim=-1)
        truncated = torch.zeros_like(probabilities).scatter(-1, indices, values)
        dropped_top_m = 1 - values.sum(dim=-1)
    source_ids = torch.as_tensor(
        artifact.source_token_ids, dtype=torch.long, device=logits.device
    )
    active = truncated.index_select(-1, source_ids)
    active_mass = active.sum(dim=-1)
    if torch.any(active_mass <= 0):
        raise SoftTransportError("selected source probability has zero active mass")
    normalized = active / active_mass.unsqueeze(-1)

    stats = SoftTransportStats(
        retained_mass=active_mass,
        dropped_top_m_mass=dropped_top_m,
        active_support_mass=probabilities.index_select(-1, source_ids).sum(dim=-1),
        top_m=top_m,
    )
    return normalized, stats


def transport_embeddings(
    logits: torch.Tensor,
    artifact: TransportArtifact,
    receiver_embedding_weight: torch.Tensor,
    *,
    tau: float,
    top_m: int | None = None,
    allow_partial_support: bool = False,
    source_vocab_size: int | None = None,
    embedding_chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, SoftTransportStats]:
    _require_torch()
    target_probabilities, stats = transport_probabilities(
        logits,
        artifact,
        tau=tau,
        top_m=top_m,
        allow_partial_support=allow_partial_support,
        source_vocab_size=source_vocab_size,
    )
    if receiver_embedding_weight.ndim != 2:
        raise SoftTransportError("receiver embedding weight must be rank two")
    if np.any(artifact.target_token_ids >= receiver_embedding_weight.shape[0]):
        raise SoftTransportError("artifact target token ID exceeds receiver vocabulary")
    if receiver_embedding_weight.device != logits.device:
        raise SoftTransportError("receiver embedding and logits must share a device")
    if (
        isinstance(embedding_chunk_size, bool)
        or not isinstance(embedding_chunk_size, int)
        or embedding_chunk_size <= 0
    ):
        raise SoftTransportError("embedding_chunk_size must be a positive integer")
    accumulator = torch.zeros(
        (*target_probabilities.shape[:-1], receiver_embedding_weight.shape[1]),
        dtype=logits.dtype,
        device=logits.device,
    )
    for start in range(0, artifact.shape[0], embedding_chunk_size):
        end = min(start + embedding_chunk_size, artifact.shape[0])
        ids_np = artifact.target_token_ids[start:end]
        first_id = int(ids_np[0])
        if np.array_equal(ids_np, np.arange(first_id, first_id + len(ids_np))):
            weight_chunk = receiver_embedding_weight[first_id : first_id + len(ids_np)]
        else:
            target_ids = torch.as_tensor(
                ids_np, dtype=torch.long, device=receiver_embedding_weight.device
            )
            weight_chunk = receiver_embedding_weight.index_select(0, target_ids)
        probability_chunk = target_probabilities[..., start:end].to(
            dtype=receiver_embedding_weight.dtype
        )
        accumulator.add_((probability_chunk @ weight_chunk).to(dtype=logits.dtype))
    embeddings = accumulator.to(dtype=receiver_embedding_weight.dtype)
    return embeddings, target_probabilities, stats
