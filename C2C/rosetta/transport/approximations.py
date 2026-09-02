"""Deterministic hard, chunked, and precomputed transport approximations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Tokenizer-only environments do not need torch.
    torch = None  # type: ignore[assignment]

from .artifact import TransportArtifact
from .soft_transport import (
    SoftTransportStats,
    source_probabilities,
    transport_probabilities,
)


class ApproximationError(ValueError):
    """Raised when an approximation input is invalid or incompatible."""


@dataclass(frozen=True)
class ApproximationErrorReport:
    cosine_similarity: torch.Tensor
    relative_l2_error: torch.Tensor


def _require_torch() -> None:
    if torch is None:
        raise ApproximationError("transport approximations require PyTorch")


def _active_receiver_embeddings(
    artifact: TransportArtifact, receiver_embedding_weight: torch.Tensor
) -> torch.Tensor:
    artifact.validate()
    if receiver_embedding_weight.ndim != 2:
        raise ApproximationError("receiver embedding weight must be rank two")
    if np.any(artifact.target_token_ids >= receiver_embedding_weight.shape[0]):
        raise ApproximationError("artifact target token ID exceeds receiver vocabulary")
    target_ids = torch.as_tensor(
        artifact.target_token_ids,
        dtype=torch.long,
        device=receiver_embedding_weight.device,
    )
    return receiver_embedding_weight.index_select(0, target_ids)


def hard_transport_embeddings(
    logits: torch.Tensor,
    artifact: TransportArtifact,
    receiver_embedding_weight: torch.Tensor,
    *,
    tau: float,
    top_m: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SoftTransportStats]:
    """Return TH embeddings, breaking probability ties by original target ID."""
    _require_torch()
    if receiver_embedding_weight.device != logits.device:
        raise ApproximationError("receiver embedding and logits must share a device")
    target, stats = transport_probabilities(logits, artifact, tau=tau, top_m=top_m)
    active_embeddings = _active_receiver_embeddings(
        artifact, receiver_embedding_weight
    ).to(dtype=logits.dtype)
    original_ids = torch.as_tensor(
        artifact.target_token_ids, dtype=torch.long, device=logits.device
    )
    maximum = target.max(dim=-1, keepdim=True).values
    sentinel = torch.iinfo(torch.long).max
    tied_ids = torch.where(target == maximum, original_ids, sentinel)
    chosen_ids = tied_ids.min(dim=-1).values
    embeddings = receiver_embedding_weight.index_select(
        0, chosen_ids.reshape(-1)
    ).reshape(*chosen_ids.shape, receiver_embedding_weight.shape[1])
    return embeddings.to(dtype=logits.dtype), chosen_ids, stats


def precompute_source_values(
    artifact: TransportArtifact,
    receiver_embedding_weight: torch.Tensor,
    *,
    edge_chunk_size: int = 65_536,
) -> torch.Tensor:
    """Materialize compact rows ``C_i = W_in^B T[:, i]`` in source order."""
    _require_torch()
    if (
        isinstance(edge_chunk_size, bool)
        or not isinstance(edge_chunk_size, int)
        or edge_chunk_size <= 0
    ):
        raise ApproximationError("edge chunk size must be a positive integer")
    active_embeddings = _active_receiver_embeddings(artifact, receiver_embedding_weight)
    source_values = torch.zeros(
        (artifact.shape[1], receiver_embedding_weight.shape[1]),
        dtype=receiver_embedding_weight.dtype,
        device=receiver_embedding_weight.device,
    )
    columns_np = np.repeat(
        np.arange(artifact.shape[1], dtype=np.int64), np.diff(artifact.indptr)
    )
    for start in range(0, len(artifact.data), edge_chunk_size):
        stop = min(start + edge_chunk_size, len(artifact.data))
        rows = torch.as_tensor(
            artifact.indices[start:stop], dtype=torch.long, device=source_values.device
        )
        columns = torch.as_tensor(
            columns_np[start:stop], dtype=torch.long, device=source_values.device
        )
        weights = torch.as_tensor(
            artifact.data[start:stop],
            dtype=source_values.dtype,
            device=source_values.device,
        )
        values = active_embeddings.index_select(0, rows) * weights.unsqueeze(-1)
        source_values.index_add_(0, columns, values)
    return source_values


def precomputed_transport_embeddings(
    logits: torch.Tensor,
    artifact: TransportArtifact,
    source_values: torch.Tensor,
    *,
    tau: float,
    top_m: int | None = None,
) -> tuple[torch.Tensor, SoftTransportStats]:
    """Apply precomputed compact source values to source probabilities."""
    _require_torch()
    if source_values.ndim != 2 or source_values.shape[0] != artifact.shape[1]:
        raise ApproximationError("source values must match artifact source columns")
    if source_values.device != logits.device:
        raise ApproximationError("source values and logits must share a device")
    source, stats = source_probabilities(logits, artifact, tau=tau, top_m=top_m)
    return source @ source_values.to(dtype=logits.dtype), stats


def chunked_transport_embeddings(
    logits: torch.Tensor,
    artifact: TransportArtifact,
    receiver_embedding_weight: torch.Tensor,
    *,
    tau: float,
    top_m: int | None = None,
    edge_chunk_size: int = 65_536,
) -> tuple[torch.Tensor, SoftTransportStats]:
    """Accumulate receiver embeddings by edge chunks without materializing C."""
    _require_torch()
    if (
        isinstance(edge_chunk_size, bool)
        or not isinstance(edge_chunk_size, int)
        or edge_chunk_size <= 0
    ):
        raise ApproximationError("edge chunk size must be a positive integer")
    if receiver_embedding_weight.device != logits.device:
        raise ApproximationError("receiver embedding and logits must share a device")
    source, stats = source_probabilities(logits, artifact, tau=tau, top_m=top_m)
    active_embeddings = _active_receiver_embeddings(
        artifact, receiver_embedding_weight
    ).to(dtype=logits.dtype)
    flat_source = source.reshape(-1, source.shape[-1])
    flat_output = torch.zeros(
        (flat_source.shape[0], active_embeddings.shape[1]),
        dtype=logits.dtype,
        device=logits.device,
    )
    columns_np = np.repeat(
        np.arange(artifact.shape[1], dtype=np.int64), np.diff(artifact.indptr)
    )
    for start in range(0, len(artifact.data), edge_chunk_size):
        stop = min(start + edge_chunk_size, len(artifact.data))
        rows = torch.as_tensor(
            artifact.indices[start:stop], dtype=torch.long, device=logits.device
        )
        columns = torch.as_tensor(
            columns_np[start:stop], dtype=torch.long, device=logits.device
        )
        weights = torch.as_tensor(
            artifact.data[start:stop], dtype=logits.dtype, device=logits.device
        )
        edge_values = active_embeddings.index_select(0, rows) * weights.unsqueeze(-1)
        flat_output += flat_source.index_select(1, columns) @ edge_values
    return flat_output.reshape(*source.shape[:-1], flat_output.shape[-1]), stats


def approximation_error(
    approximate: torch.Tensor, exact: torch.Tensor
) -> ApproximationErrorReport:
    """Return zero-safe cosine and relative L2 error per final-dimension row."""
    _require_torch()
    if approximate.shape != exact.shape or approximate.ndim < 1:
        raise ApproximationError(
            "approximate and exact tensors must share a nonempty shape"
        )
    if not approximate.is_floating_point() or not exact.is_floating_point():
        raise ApproximationError("approximation error requires floating tensors")
    if not torch.isfinite(approximate).all() or not torch.isfinite(exact).all():
        raise ApproximationError("approximation error inputs must be finite")
    approx = approximate.float()
    reference = exact.float()
    approx_norm = torch.linalg.vector_norm(approx, dim=-1)
    reference_norm = torch.linalg.vector_norm(reference, dim=-1)
    both_zero = (approx_norm == 0) & (reference_norm == 0)
    one_zero = (approx_norm == 0) | (reference_norm == 0)
    denominator = approx_norm * reference_norm
    cosine = torch.where(
        both_zero,
        torch.ones_like(denominator),
        torch.where(
            one_zero,
            torch.zeros_like(denominator),
            (approx * reference).sum(dim=-1) / denominator,
        ),
    )
    difference = torch.linalg.vector_norm(approx - reference, dim=-1)
    relative = torch.where(
        both_zero,
        torch.zeros_like(reference_norm),
        torch.where(
            reference_norm == 0,
            torch.full_like(reference_norm, float("inf")),
            difference / reference_norm,
        ),
    )
    return ApproximationErrorReport(cosine, relative)
