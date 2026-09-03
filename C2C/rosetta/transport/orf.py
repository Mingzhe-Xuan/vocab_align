"""Positive block-orthogonal random features for soft-token transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # Tokenizer-only environments do not need torch.
    torch = None  # type: ignore[assignment]

from .artifact import TransportArtifact


class OrfError(ValueError):
    """Raised when ORF construction or application is invalid."""


@dataclass(frozen=True)
class OrfTransportState:
    omega: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    tau: float
    seed: int
    source_fingerprint: str
    target_fingerprint: str
    source_vocab_size: int
    target_vocab_size: int

    @property
    def feature_count(self) -> int:
        return int(self.omega.shape[0])

    @property
    def memory_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.omega, self.numerator, self.denominator)
        )


def _require_torch() -> None:
    if torch is None:
        raise OrfError("ORF transport requires PyTorch")


def build_orf_directions(
    feature_count: int,
    dimension: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build seeded Gaussian-radius, block-orthogonal row directions."""
    _require_torch()
    if (
        isinstance(feature_count, bool)
        or not isinstance(feature_count, int)
        or feature_count <= 0
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
    ):
        raise OrfError("feature count and dimension must be positive integers")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise OrfError("ORF seed must be a nonnegative integer")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    blocks = []
    remaining = feature_count
    while remaining:
        gaussian = torch.randn(
            (dimension, dimension),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        q, r = torch.linalg.qr(gaussian)
        signs = torch.sign(torch.diagonal(r))
        signs[signs == 0] = 1
        q = q * signs.unsqueeze(0)
        radii = gaussian.norm(dim=0)
        block = radii.unsqueeze(1) * q.T
        blocks.append(block[:remaining])
        remaining -= min(remaining, dimension)
    return torch.cat(blocks, dim=0)


def positive_orf_features(
    values: torch.Tensor,
    omega: torch.Tensor,
    *,
    stabilize: bool,
) -> torch.Tensor:
    """Evaluate nonnegative ORF features for batched row vectors."""
    _require_torch()
    if values.ndim < 1 or omega.ndim != 2 or values.shape[-1] != omega.shape[1]:
        raise OrfError("ORF value and direction dimensions do not match")
    if not values.is_floating_point() or not torch.isfinite(values).all():
        raise OrfError("ORF values must be finite floating tensors")
    if not torch.isfinite(omega).all():
        raise OrfError("ORF directions must be finite")
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1]).float()
    log_features = flat @ omega.T - 0.5 * flat.square().sum(dim=-1, keepdim=True)
    if stabilize:
        log_features -= log_features.max(dim=-1, keepdim=True).values
    features = torch.exp(log_features) / np.sqrt(omega.shape[0])
    if not torch.isfinite(features).all():
        raise OrfError("ORF features became non-finite")
    return features.reshape(*original_shape, omega.shape[0])


def _source_value_block(
    artifact: TransportArtifact,
    active_receiver_embeddings: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    values = torch.zeros(
        (stop - start, active_receiver_embeddings.shape[1]),
        dtype=torch.float32,
        device=active_receiver_embeddings.device,
    )
    edge_start = int(artifact.indptr[start])
    edge_stop = int(artifact.indptr[stop])
    counts = np.diff(artifact.indptr[start : stop + 1])
    local_columns = torch.as_tensor(
        np.repeat(np.arange(stop - start, dtype=np.int64), counts),
        dtype=torch.long,
        device=values.device,
    )
    rows = torch.as_tensor(
        artifact.indices[edge_start:edge_stop],
        dtype=torch.long,
        device=values.device,
    )
    weights = torch.as_tensor(
        artifact.data[edge_start:edge_stop],
        dtype=torch.float32,
        device=values.device,
    )
    contributions = active_receiver_embeddings.index_select(0, rows).float()
    contributions = contributions * weights.unsqueeze(-1)
    values.index_add_(0, local_columns, contributions)
    return values


def build_orf_transport_state(
    output_weight: torch.Tensor,
    output_bias: torch.Tensor | None,
    artifact: TransportArtifact,
    receiver_embedding_weight: torch.Tensor,
    *,
    feature_count: int,
    tau: float,
    seed: int,
    source_chunk_size: int = 1_024,
    source_vocab_size: int | None = None,
) -> OrfTransportState:
    """Pre-aggregate sparse-transport ORF numerator ``S`` and denominator ``z``."""
    _require_torch()
    artifact.validate()
    if output_weight.ndim != 2 or receiver_embedding_weight.ndim != 2:
        raise OrfError("output and receiver embedding weights must be rank two")
    if output_weight.device != receiver_embedding_weight.device:
        raise OrfError("output and receiver weights must share a device")
    if not np.isfinite(tau) or tau <= 0:
        raise OrfError("ORF tau must be finite and positive")
    if (
        isinstance(source_chunk_size, bool)
        or not isinstance(source_chunk_size, int)
        or source_chunk_size <= 0
    ):
        raise OrfError("source chunk size must be a positive integer")
    if np.any(artifact.source_token_ids >= output_weight.shape[0]):
        raise OrfError("artifact source token ID exceeds output vocabulary")
    if source_vocab_size is None:
        source_vocab_size = output_weight.shape[0]
    if (
        isinstance(source_vocab_size, bool)
        or not isinstance(source_vocab_size, int)
        or not 1 <= source_vocab_size <= output_weight.shape[0]
    ):
        raise OrfError("source_vocab_size must be in the output vocabulary")
    if not np.array_equal(
        artifact.source_token_ids,
        np.arange(source_vocab_size, dtype=np.int64),
    ):
        raise OrfError(
            "artifact must provide contiguous full support for source_vocab_size"
        )
    if np.any(artifact.target_token_ids >= receiver_embedding_weight.shape[0]):
        raise OrfError("artifact target token ID exceeds receiver vocabulary")
    if output_bias is not None and output_bias.shape != (output_weight.shape[0],):
        raise OrfError("output bias must have one value per source vocabulary item")
    if (
        not torch.isfinite(output_weight).all()
        or not torch.isfinite(receiver_embedding_weight).all()
    ):
        raise OrfError("ORF model weights must be finite")
    if output_bias is not None and not torch.isfinite(output_bias).all():
        raise OrfError("ORF output bias must be finite")

    device = output_weight.device
    omega = build_orf_directions(
        feature_count, output_weight.shape[1], seed=seed, device=device
    )
    target_ids = torch.as_tensor(
        artifact.target_token_ids, dtype=torch.long, device=device
    )
    active_receiver = receiver_embedding_weight.index_select(0, target_ids)
    numerator = torch.zeros(
        (receiver_embedding_weight.shape[1], feature_count),
        dtype=torch.float32,
        device=device,
    )
    denominator = torch.zeros(feature_count, dtype=torch.float32, device=device)
    source_ids_all = torch.as_tensor(
        artifact.source_token_ids, dtype=torch.long, device=device
    )
    if output_bias is None:
        bias = torch.zeros(output_weight.shape[0], dtype=torch.float32, device=device)
    else:
        bias = output_bias.detach().to(device=device, dtype=torch.float32)
    bias_shift = bias.index_select(0, source_ids_all).max()

    for start in range(0, artifact.shape[1], source_chunk_size):
        stop = min(start + source_chunk_size, artifact.shape[1])
        source_ids = source_ids_all[start:stop]
        keys = output_weight.index_select(0, source_ids)
        key_features = positive_orf_features(keys, omega, stabilize=False)
        alpha = torch.exp(
            (bias.index_select(0, source_ids) - bias_shift) / tau
        ).unsqueeze(1)
        weighted_features = alpha * key_features
        if not torch.isfinite(weighted_features).all():
            raise OrfError("ORF bias-weighted key features became non-finite")
        source_values = _source_value_block(artifact, active_receiver, start, stop)
        numerator += source_values.T @ weighted_features
        denominator += weighted_features.sum(dim=0)
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all():
        raise OrfError("ORF pre-aggregation became non-finite")
    if torch.any(denominator <= 0):
        raise OrfError("ORF pre-aggregation denominator must be positive")
    return OrfTransportState(
        omega=omega,
        numerator=numerator,
        denominator=denominator,
        tau=float(tau),
        seed=seed,
        source_fingerprint=str(artifact.metadata["source_fingerprint"]),
        target_fingerprint=str(artifact.metadata["target_fingerprint"]),
        source_vocab_size=source_vocab_size,
        target_vocab_size=receiver_embedding_weight.shape[0],
    )


def apply_orf_transport(
    hidden: torch.Tensor,
    state: OrfTransportState,
    *,
    source_fingerprint: str | None = None,
    target_fingerprint: str | None = None,
) -> torch.Tensor:
    """Apply ``u @ S.T / (u @ z)`` to batched source hidden row vectors."""
    _require_torch()
    if (
        source_fingerprint is not None
        and source_fingerprint != state.source_fingerprint
    ):
        raise OrfError("ORF source fingerprint mismatch")
    if (
        target_fingerprint is not None
        and target_fingerprint != state.target_fingerprint
    ):
        raise OrfError("ORF target fingerprint mismatch")
    if hidden.shape[-1] != state.omega.shape[1]:
        raise OrfError("ORF hidden dimension mismatch")
    if hidden.device != state.omega.device:
        raise OrfError("ORF hidden state and precomputed state must share a device")
    original_dtype = hidden.dtype
    features = positive_orf_features(
        hidden.float() / state.tau, state.omega, stabilize=True
    )
    denominator = features @ state.denominator
    threshold = torch.finfo(denominator.dtype).eps
    if not torch.isfinite(denominator).all() or torch.any(denominator <= threshold):
        raise OrfError("ORF online denominator is non-positive or non-finite")
    result = (features @ state.numerator.T) / denominator.unsqueeze(-1)
    if not torch.isfinite(result).all():
        raise OrfError("ORF transport output became non-finite")
    return result.to(dtype=original_dtype)
