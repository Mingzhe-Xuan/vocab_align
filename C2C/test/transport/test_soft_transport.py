import numpy as np
import pytest
import torch
from dataclasses import replace

from rosetta.transport.artifact import artifact_from_dense
from rosetta.transport.soft_transport import (
    SoftTransportError,
    transport_embeddings,
    transport_probabilities,
)


def _artifact(transport, source, target):
    return artifact_from_dense(
        np.asarray(transport, dtype=np.float64),
        np.asarray(source, dtype=np.float64),
        np.asarray(target, dtype=np.float64),
        {
            "schema_version": 1,
            "source_fingerprint": "source",
            "target_fingerprint": "target",
            "build_config": {"epsilon": 0.5},
            "seed": 42,
            "code_version": "test",
        },
    )


def test_sparse_transport_and_embeddings_match_dense_oracles():
    transport = np.array([[1.0, 0.25, 0.0], [0.0, 0.75, 1.0]])
    source = np.array([0.2, 0.4, 0.4])
    artifact = _artifact(transport, source, transport @ source)
    logits = torch.tensor([[[0.1, 0.2, 0.3], [2.0, -1.0, 0.5]]], dtype=torch.float64)
    receiver = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64)
    probabilities, stats = transport_probabilities(logits, artifact, tau=0.7)
    dense_source = torch.softmax(logits / 0.7, dim=-1)
    expected = torch.as_tensor(transport) @ dense_source.unsqueeze(-1)
    torch.testing.assert_close(probabilities, expected.squeeze(-1))
    embeddings, via_embeddings, _ = transport_embeddings(
        logits, artifact, receiver, tau=0.7
    )
    torch.testing.assert_close(via_embeddings, probabilities)
    torch.testing.assert_close(embeddings, probabilities @ receiver)
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones((1, 2), dtype=torch.float64))
    assert embeddings.dtype == logits.dtype
    assert stats.retained_mass.shape == logits.shape[:-1]


def test_top_m_full_is_exact_and_discarded_mass_is_monotone():
    artifact = _artifact(np.eye(4), np.full(4, 0.25), np.full(4, 0.25))
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    exact, _ = transport_probabilities(logits, artifact, tau=1.0)
    full, full_stats = transport_probabilities(logits, artifact, tau=1.0, top_m=4)
    one, one_stats = transport_probabilities(logits, artifact, tau=1.0, top_m=1)
    two, two_stats = transport_probabilities(logits, artifact, tau=1.0, top_m=2)
    torch.testing.assert_close(full, exact)
    assert torch.all(one_stats.dropped_top_m_mass >= two_stats.dropped_top_m_mass)
    assert torch.all(two_stats.dropped_top_m_mass >= full_stats.dropped_top_m_mass)
    torch.testing.assert_close(one.sum(-1), torch.ones(1))


def test_invalid_temperature_vocab_and_partial_support_fail():
    artifact = _artifact(np.eye(2), [0.5, 0.5], [0.5, 0.5])
    logits = torch.zeros(1, 2)
    with pytest.raises(SoftTransportError, match="tau"):
        transport_probabilities(logits, artifact, tau=0)
    with pytest.raises(SoftTransportError, match="top_m"):
        transport_probabilities(logits, artifact, tau=1, top_m=3)
    with pytest.raises(SoftTransportError, match="exceeds"):
        transport_probabilities(torch.zeros(1, 1), artifact, tau=1)
    partial = _artifact(np.ones((1, 1)), [1.0], [1.0])
    with pytest.raises(SoftTransportError, match="full source"):
        transport_probabilities(torch.zeros(1, 2), partial, tau=1)


def test_original_token_id_mappings_control_gather_and_receiver_embedding():
    artifact = _artifact(np.eye(2), [0.5, 0.5], [0.5, 0.5])
    artifact = replace(
        artifact,
        source_token_ids=np.array([1, 0]),
        target_token_ids=np.array([2, 0]),
    )
    logits = torch.tensor([[0.0, 2.0]])
    receiver = torch.tensor([[10.0], [20.0], [30.0]])
    embeddings, target, _ = transport_embeddings(
        logits, artifact, receiver, tau=1.0
    )
    source = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(target, source[:, [1, 0]])
    expected = source[:, 1] * 30.0 + source[:, 0] * 10.0
    torch.testing.assert_close(embeddings[:, 0], expected)
