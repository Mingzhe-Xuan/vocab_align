import numpy as np
import pytest
import torch

from rosetta.transport.artifact import artifact_from_dense
from rosetta.transport.orf import (
    OrfError,
    apply_orf_transport,
    build_orf_directions,
    build_orf_transport_state,
    positive_orf_features,
)


def _artifact(transport):
    source = np.full(transport.shape[1], 1 / transport.shape[1])
    target = transport @ source
    return artifact_from_dense(
        transport,
        source,
        target,
        {
            "schema_version": 1,
            "source_fingerprint": "source-fp",
            "target_fingerprint": "target-fp",
            "build_config": {"epsilon": 0.5},
            "seed": 42,
            "code_version": "test",
        },
    )


def test_orf_directions_are_seeded_and_block_orthogonal():
    first = build_orf_directions(7, 3, seed=42)
    second = build_orf_directions(7, 3, seed=42)
    other = build_orf_directions(7, 3, seed=43)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, other)
    assert first.shape == (7, 3)
    normalized = first[:3] / first[:3].norm(dim=1, keepdim=True)
    torch.testing.assert_close(
        normalized @ normalized.T, torch.eye(3), atol=1e-6, rtol=0
    )


def test_sparse_orf_preaggregation_matches_manual_row_vector_formula():
    transport = np.array([[0.8, 0.1, 0.0], [0.2, 0.9, 1.0]], dtype=np.float64)
    artifact = _artifact(transport)
    output_weight = torch.tensor(
        [[0.10, 0.20], [-0.20, 0.15], [0.05, -0.10]], dtype=torch.float32
    )
    output_bias = torch.tensor([0.3, -0.1, 0.2])
    receiver = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    state = build_orf_transport_state(
        output_weight,
        output_bias,
        artifact,
        receiver,
        feature_count=16,
        tau=0.7,
        seed=9,
        source_chunk_size=1,
    )
    values = torch.as_tensor(transport.T, dtype=torch.float32) @ receiver
    keys = positive_orf_features(output_weight, state.omega, stabilize=False)
    alpha = torch.exp((output_bias - output_bias.max()) / 0.7).unsqueeze(1)
    weighted = alpha * keys
    torch.testing.assert_close(state.numerator, values.T @ weighted)
    torch.testing.assert_close(state.denominator, weighted.sum(0))

    hidden = torch.tensor([[[0.2, -0.1], [0.0, 0.3]]])
    query = positive_orf_features(hidden / 0.7, state.omega, stabilize=True)
    expected = (query @ state.numerator.T) / (query @ state.denominator).unsqueeze(-1)
    actual = apply_orf_transport(
        hidden,
        state,
        source_fingerprint="source-fp",
        target_fingerprint="target-fp",
    )
    torch.testing.assert_close(actual, expected)
    assert state.memory_bytes == sum(
        tensor.numel() * tensor.element_size()
        for tensor in (state.omega, state.numerator, state.denominator)
    )


def test_orf_feature_count_changes_shape_memory_and_is_reproducible():
    artifact = _artifact(np.eye(2))
    output = torch.tensor([[0.1, 0.0], [0.0, 0.1]])
    receiver = torch.eye(2)
    small = build_orf_transport_state(
        output, None, artifact, receiver, feature_count=4, tau=1.0, seed=5
    )
    repeat = build_orf_transport_state(
        output, None, artifact, receiver, feature_count=4, tau=1.0, seed=5
    )
    large = build_orf_transport_state(
        output, None, artifact, receiver, feature_count=8, tau=1.0, seed=5
    )
    torch.testing.assert_close(small.omega, repeat.omega, rtol=0, atol=0)
    torch.testing.assert_close(small.numerator, repeat.numerator, rtol=0, atol=0)
    assert small.numerator.shape == (2, 4)
    assert large.numerator.shape == (2, 8)
    assert large.memory_bytes > small.memory_bytes


def test_orf_validation_and_fingerprint_failures_are_explicit():
    with pytest.raises(OrfError, match="positive"):
        build_orf_directions(0, 2, seed=1)
    artifact = _artifact(np.eye(2))
    output = torch.eye(2)
    receiver = torch.eye(2)
    with pytest.raises(OrfError, match="tau"):
        build_orf_transport_state(
            output, None, artifact, receiver, feature_count=2, tau=0, seed=1
        )
    state = build_orf_transport_state(
        output, None, artifact, receiver, feature_count=2, tau=1, seed=1
    )
    with pytest.raises(OrfError, match="fingerprint"):
        apply_orf_transport(torch.zeros(1, 2), state, source_fingerprint="wrong")
    broken = state.__class__(
        **{**state.__dict__, "denominator": torch.zeros_like(state.denominator)}
    )
    with pytest.raises(OrfError, match="denominator"):
        apply_orf_transport(torch.zeros(1, 2), broken)
