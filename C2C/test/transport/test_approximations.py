import numpy as np
import pytest
import torch

from rosetta.transport.approximations import (
    ApproximationError,
    approximation_error,
    chunked_transport_embeddings,
    hard_transport_embeddings,
    precompute_source_values,
    precomputed_transport_embeddings,
)
from rosetta.transport.artifact import artifact_from_dense
from rosetta.transport.soft_transport import transport_embeddings


def _artifact(transport):
    source = np.full(transport.shape[1], 1 / transport.shape[1])
    target = transport @ source
    return artifact_from_dense(
        transport,
        source,
        target,
        {
            "schema_version": 1,
            "source_fingerprint": "source",
            "target_fingerprint": "target",
            "build_config": {"epsilon": 0.5},
            "seed": 42,
            "code_version": "test",
        },
    )


def test_chunked_and_precomputed_embeddings_match_exact_sparse_oracle():
    transport = np.array(
        [
            [0.8, 0.0, 0.1, 0.2],
            [0.2, 0.5, 0.0, 0.3],
            [0.0, 0.5, 0.9, 0.5],
        ],
        dtype=np.float64,
    )
    artifact = _artifact(transport)
    logits = torch.tensor(
        [[[0.1, 0.2, 0.3, 0.4], [2.0, -1.0, 0.0, 0.5]]],
        dtype=torch.float64,
    )
    receiver = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.5]], dtype=torch.float64)
    exact, _, _ = transport_embeddings(logits, artifact, receiver, tau=0.7)
    source_values = precompute_source_values(artifact, receiver, edge_chunk_size=2)
    precomputed, _ = precomputed_transport_embeddings(
        logits, artifact, source_values, tau=0.7
    )
    chunked, _ = chunked_transport_embeddings(
        logits, artifact, receiver, tau=0.7, edge_chunk_size=2
    )
    torch.testing.assert_close(source_values, torch.as_tensor(transport.T) @ receiver)
    torch.testing.assert_close(precomputed, exact)
    torch.testing.assert_close(chunked, exact)


def test_hard_transport_uses_smallest_original_target_id_for_ties():
    artifact = _artifact(np.full((2, 2), 0.5))
    artifact = artifact.__class__(
        **{
            **artifact.__dict__,
            "target_token_ids": np.array([3, 1], dtype=np.int64),
        }
    )
    receiver = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    embeddings, ids, _ = hard_transport_embeddings(
        torch.zeros(2, 2), artifact, receiver, tau=1.0
    )
    torch.testing.assert_close(ids, torch.ones(2, dtype=torch.long))
    torch.testing.assert_close(embeddings, receiver[1].expand(2, -1))


def test_top_m_paths_and_zero_safe_error_semantics():
    artifact = _artifact(np.eye(4))
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    receiver = torch.eye(4)
    source_values = precompute_source_values(artifact, receiver)
    full, full_stats = precomputed_transport_embeddings(
        logits, artifact, source_values, tau=1.0, top_m=4
    )
    exact, _, exact_stats = transport_embeddings(logits, artifact, receiver, tau=1.0)
    one, one_stats = chunked_transport_embeddings(
        logits, artifact, receiver, tau=1.0, top_m=1, edge_chunk_size=1
    )
    torch.testing.assert_close(full, exact)
    assert torch.all(one_stats.dropped_top_m_mass >= full_stats.dropped_top_m_mass)
    torch.testing.assert_close(exact_stats.dropped_top_m_mass, torch.zeros(1))
    assert one.shape == exact.shape

    report = approximation_error(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
    )
    torch.testing.assert_close(report.cosine_similarity, torch.tensor([1.0, 0.0, 0.0]))
    assert report.relative_l2_error[0] == 0
    assert torch.isinf(report.relative_l2_error[1])
    torch.testing.assert_close(
        report.relative_l2_error[2], torch.sqrt(torch.tensor(2.0))
    )
    assert not torch.isnan(report.cosine_similarity).any()
    assert not torch.isnan(report.relative_l2_error).any()


def test_approximation_validation_is_explicit():
    artifact = _artifact(np.eye(2))
    receiver = torch.eye(2)
    with pytest.raises(ApproximationError, match="chunk"):
        precompute_source_values(artifact, receiver, edge_chunk_size=0)
    with pytest.raises(ApproximationError, match="source values"):
        precomputed_transport_embeddings(
            torch.zeros(1, 2), artifact, torch.zeros(1, 2), tau=1.0
        )
    with pytest.raises(ApproximationError, match="finite"):
        approximation_error(torch.tensor([[float("nan")]]), torch.zeros(1, 1))


def test_all_logit_approximations_ignore_verified_padded_lm_head_rows():
    artifact = _artifact(np.eye(3))
    logits = torch.tensor([[0.0, 2.0, 1.0, 100.0]])
    receiver = torch.eye(3)
    source_values = precompute_source_values(artifact, receiver)
    exact, _, _ = transport_embeddings(
        logits, artifact, receiver, tau=1.0, source_vocab_size=3
    )
    hard, chosen, _ = hard_transport_embeddings(
        logits, artifact, receiver, tau=1.0, source_vocab_size=3
    )
    precomputed, _ = precomputed_transport_embeddings(
        logits,
        artifact,
        source_values,
        tau=1.0,
        source_vocab_size=3,
    )
    chunked, _ = chunked_transport_embeddings(
        logits,
        artifact,
        receiver,
        tau=1.0,
        edge_chunk_size=1,
        source_vocab_size=3,
    )
    torch.testing.assert_close(precomputed, exact)
    torch.testing.assert_close(chunked, exact)
    torch.testing.assert_close(chosen, torch.tensor([1]))
    torch.testing.assert_close(hard, receiver[1].unsqueeze(0))
