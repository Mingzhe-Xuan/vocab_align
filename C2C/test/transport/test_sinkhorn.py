import numpy as np
import pytest

from rosetta.transport.sinkhorn import (
    SinkhornError,
    conditional_from_coupling,
    dense_sinkhorn,
)


@pytest.mark.parametrize("shape", [(2, 3), (3, 2)])
def test_dense_sinkhorn_non_square_marginals(shape):
    cost = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 5
    source = np.full(shape[1], 1 / shape[1])
    target = np.full(shape[0], 1 / shape[0])
    coupling, report = dense_sinkhorn(
        cost, source, target, epsilon=0.7, tolerance=1e-10
    )

    assert report.converged
    assert report.iterations <= report.max_iter
    np.testing.assert_allclose(coupling.sum(axis=0), source, atol=1e-10)
    np.testing.assert_allclose(coupling.sum(axis=1), target, atol=1e-10)
    transport = conditional_from_coupling(coupling, source)
    np.testing.assert_allclose(transport.sum(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(transport @ source, target, atol=1e-10)


def test_dense_sinkhorn_preserves_missing_edges_at_small_epsilon():
    cost = np.array([[0.0, np.inf], [np.inf, 0.0]])
    coupling, _ = dense_sinkhorn(
        cost,
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        epsilon=1e-6,
    )
    assert coupling[0, 1] == 0.0
    assert coupling[1, 0] == 0.0
    assert np.all(np.isfinite(coupling))


def test_dense_sinkhorn_rejects_infeasible_or_unconverged_problem():
    with pytest.raises(SinkhornError, match="target token"):
        dense_sinkhorn(
            np.array([[np.inf, np.inf], [0.0, 0.0]]),
            np.array([0.5, 0.5]),
            np.array([0.5, 0.5]),
            epsilon=1.0,
        )
    with pytest.raises(SinkhornError, match="did not converge"):
        dense_sinkhorn(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0.9, 0.1]),
            np.array([0.2, 0.8]),
            epsilon=0.01,
            tolerance=1e-15,
            max_iter=1,
        )
