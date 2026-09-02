import numpy as np
import pytest
from types import SimpleNamespace

from rosetta.transport.candidate_graph import (
    CandidateEdge,
    CandidateGraph,
    CandidateGraphError,
    EdgeSource,
    augment_candidate_graph_for_marginals,
)
from rosetta.transport.sinkhorn import (
    SinkhornError,
    _dual_increment_value_gradient,
    _dual_value_gradient,
    candidate_edge_costs,
    dense_sinkhorn,
    sparse_conditional_from_coupling,
    sparse_log_sinkhorn,
)


def test_sparse_dual_gradient_matches_finite_difference():
    rows = np.repeat(np.arange(2), 3)
    columns = np.tile(np.arange(3), 2)
    log_kernel = np.log(np.array([0.7, 0.2, 0.1, 0.1, 0.3, 0.6]))
    source = np.array([0.2, 0.3, 0.5])
    target = np.array([0.4, 0.6])
    variables = np.array([0.2, -0.1, 0.1, -0.2])
    value, gradient, _ = _dual_value_gradient(
        variables, rows, columns, log_kernel, source, target
    )
    numerical = np.empty_like(gradient)
    step = 1e-6
    for index in range(len(variables)):
        offset = np.zeros_like(variables)
        offset[index] = step
        upper, _, _ = _dual_value_gradient(
            variables + offset, rows, columns, log_kernel, source, target
        )
        lower, _, _ = _dual_value_gradient(
            variables - offset, rows, columns, log_kernel, source, target
        )
        numerical[index] = (upper - lower) / (2 * step)
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, atol=1e-9)


def test_sparse_dual_increment_gradient_is_stable_around_large_baseline():
    rows = np.repeat(np.arange(2), 3)
    columns = np.tile(np.arange(3), 2)
    log_kernel = np.log(np.array([0.7, 0.2, 0.1, 0.1, 0.3, 0.6]))
    log_kernel[columns == 2] -= 120.0
    source = np.array([0.2, 0.3, 0.5])
    target = np.array([0.4, 0.6])
    base = np.array([120.2, 119.9, -120.0, -120.3])
    increments = np.array([2e-5, -1e-5, 1e-5, -2e-5])
    value, gradient, _ = _dual_increment_value_gradient(
        increments, base, rows, columns, log_kernel, source, target
    )
    numerical = np.empty_like(gradient)
    step = 1e-7
    for index in range(len(increments)):
        offset = np.zeros_like(increments)
        offset[index] = step
        upper, _, _ = _dual_increment_value_gradient(
            increments + offset,
            base,
            rows,
            columns,
            log_kernel,
            source,
            target,
        )
        lower, _, _ = _dual_increment_value_gradient(
            increments - offset,
            base,
            rows,
            columns,
            log_kernel,
            source,
            target,
        )
        numerical[index] = (upper - lower) / (2 * step)
    assert np.isfinite(value)
    np.testing.assert_allclose(gradient, numerical, atol=1e-8)


def test_dual_acceleration_converges_on_pathological_sparse_scaling():
    size = 80
    graph = CandidateGraph(
        size,
        size,
        tuple(
            [CandidateEdge(i, i, EdgeSource.EXACT_BYTE, 1.0) for i in range(size)]
            + [CandidateEdge(i, 0, EdgeSource.ANN, 1e-6) for i in range(1, size)]
            + [CandidateEdge(0, j, EdgeSource.ANN, 1e-6) for j in range(1, size)]
        ),
    )
    source = np.geomspace(1.0, 1e-14, size)
    source /= source.sum()
    target = source[::-1].copy()
    graph, _ = augment_candidate_graph_for_marginals(graph, source, target)
    with pytest.raises(SinkhornError, match="did not converge"):
        sparse_log_sinkhorn(
            graph,
            source,
            target,
            epsilon=0.5,
            tolerance=1e-9,
            max_iter=100,
            acceleration_after=None,
        )
    coupling, report = sparse_log_sinkhorn(
        graph,
        source,
        target,
        epsilon=0.5,
        tolerance=1e-9,
        max_iter=1_500,
        acceleration_after=10,
        acceleration_max_evaluations=800,
    )
    assert report.converged
    assert report.method == "sinkhorn-scaled-lbfgs-sinkhorn"
    assert 0 < report.acceleration_evaluations < 400
    assert report.iterations <= report.max_iter
    np.testing.assert_allclose(coupling.to_dense().sum(axis=0), source, atol=1e-9)
    np.testing.assert_allclose(coupling.to_dense().sum(axis=1), target, atol=1e-9)


def test_dual_acceleration_forwards_bounded_lbfgs_workspace(monkeypatch):
    captured = []

    def fake_minimize(objective, initial, *, method, jac, options):
        value, gradient = objective(initial)
        assert np.isfinite(value)
        assert np.all(np.isfinite(gradient))
        captured.append({"method": method, "jac": jac, "options": options})
        return SimpleNamespace(x=initial, nfev=1)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
    graph = CandidateGraph(
        2,
        2,
        (
            CandidateEdge(0, 0, EdgeSource.EXACT_BYTE, 1.0),
            CandidateEdge(0, 1, EdgeSource.ANN, 1e-6),
            CandidateEdge(1, 0, EdgeSource.ANN, 1e-6),
            CandidateEdge(1, 1, EdgeSource.EXACT_BYTE, 1.0),
        ),
    )
    source = np.array([0.8, 0.2])
    target = np.array([0.2, 0.8])
    sparse_log_sinkhorn(
        graph,
        source,
        target,
        epsilon=0.5,
        tolerance=1e-9,
        max_iter=100,
        acceleration_after=1,
        acceleration_max_evaluations=2,
        acceleration_history_size=2,
    )
    assert captured[0]["method"] == "L-BFGS-B"
    assert captured[0]["jac"] is True
    assert captured[0]["options"]["maxcor"] == 2
    assert captured[0]["options"]["maxfun"] == 2
    assert captured[0]["options"]["ftol"] == 0
    assert all(item["options"]["maxfun"] <= 2 for item in captured)
    with pytest.raises(SinkhornError, match="history size"):
        sparse_log_sinkhorn(
            graph,
            source,
            target,
            epsilon=0.5,
            acceleration_history_size=0,
        )


def test_dual_acceleration_scaled_gradient_matches_finite_difference(monkeypatch):
    checked = False

    def fake_minimize(objective, initial, *, method, jac, options):
        nonlocal checked
        value, gradient = objective(initial)
        numerical = np.empty_like(gradient)
        step = 1e-6
        for index in range(len(initial)):
            offset = np.zeros_like(initial)
            offset[index] = step
            upper, _ = objective(initial + offset)
            lower, _ = objective(initial - offset)
            numerical[index] = (upper - lower) / (2 * step)
        assert np.isfinite(value)
        np.testing.assert_allclose(gradient, numerical, atol=1e-8)
        checked = True
        return SimpleNamespace(x=initial, nfev=1)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
    graph = _complete_graph(2, 3)
    sparse_log_sinkhorn(
        graph,
        np.array([1e-8, 0.2, 0.8 - 1e-8]),
        np.array([0.3, 0.7]),
        epsilon=0.5,
        tolerance=1e-9,
        max_iter=100,
        acceleration_after=1,
        acceleration_max_evaluations=32,
    )
    assert checked


def test_dual_acceleration_restarts_after_early_unimproved_termination(monkeypatch):
    import scipy.optimize

    real_minimize = scipy.optimize.minimize
    calls = 0

    def short_then_real(objective, initial, *, method, jac, options):
        nonlocal calls
        calls += 1
        if calls == 1:
            objective(initial)
            return SimpleNamespace(x=initial, status=2, message="forced short return")
        return real_minimize(
            objective, initial, method=method, jac=jac, options=options
        )

    monkeypatch.setattr("scipy.optimize.minimize", short_then_real)
    size = 80
    graph = CandidateGraph(
        size,
        size,
        tuple(
            [CandidateEdge(i, i, EdgeSource.EXACT_BYTE, 1.0) for i in range(size)]
            + [CandidateEdge(i, 0, EdgeSource.ANN, 1e-6) for i in range(1, size)]
            + [CandidateEdge(0, j, EdgeSource.ANN, 1e-6) for j in range(1, size)]
        ),
    )
    source = np.geomspace(1.0, 1e-14, size)
    source /= source.sum()
    target = source[::-1].copy()
    graph, _ = augment_candidate_graph_for_marginals(graph, source, target)
    coupling, report = sparse_log_sinkhorn(
        graph,
        source,
        target,
        epsilon=0.5,
        tolerance=1e-9,
        max_iter=1_500,
        acceleration_after=10,
        acceleration_max_evaluations=800,
    )
    assert calls >= 2
    assert report.acceleration_attempts >= 2
    assert "forced short return" in report.acceleration_terminations[0]
    assert report.acceleration_evaluations <= 800
    np.testing.assert_allclose(coupling.to_dense().sum(axis=0), source, atol=1e-9)
    np.testing.assert_allclose(coupling.to_dense().sum(axis=1), target, atol=1e-9)


def test_marginal_augmentation_repairs_connected_capacity_infeasibility():
    graph = CandidateGraph(
        3,
        3,
        (
            CandidateEdge(0, 0, EdgeSource.EXACT_BYTE, 1.0),
            CandidateEdge(1, 0, EdgeSource.ANN, 0.5),
            CandidateEdge(1, 1, EdgeSource.EXACT_BYTE, 1.0),
            CandidateEdge(1, 2, EdgeSource.ANN, 0.5),
            CandidateEdge(2, 2, EdgeSource.EXACT_BYTE, 1.0),
        ),
    )
    source = np.array([0.8, 0.1, 0.1])
    target = np.array([0.1, 0.1, 0.8])
    with pytest.raises(SinkhornError, match="did not converge"):
        sparse_log_sinkhorn(
            graph, source, target, epsilon=0.5, tolerance=1e-9, max_iter=100
        )

    augmented, added = augment_candidate_graph_for_marginals(graph, source, target)
    assert 0 < added <= len(source) + len(target) - 1
    assert (
        sum(edge.source == EdgeSource.FEASIBILITY for edge in augmented.edges) == added
    )
    coupling, report = sparse_log_sinkhorn(
        augmented, source, target, epsilon=0.5, tolerance=1e-9, max_iter=10_000
    )
    dense = coupling.to_dense()
    np.testing.assert_allclose(dense.sum(axis=0), source, atol=1e-9)
    np.testing.assert_allclose(dense.sum(axis=1), target, atol=1e-9)
    assert report.converged


def test_marginal_augmentation_preserves_already_feasible_support_and_validates():
    graph = CandidateGraph(
        2,
        2,
        (
            CandidateEdge(0, 0, EdgeSource.EXACT_BYTE, 1.0),
            CandidateEdge(1, 1, EdgeSource.EXACT_BYTE, 1.0),
        ),
    )
    source = np.array([0.5, 0.5])
    target = np.array([0.5, 0.5])
    unchanged, added = augment_candidate_graph_for_marginals(graph, source, target)
    assert unchanged is graph
    assert added == 0
    with pytest.raises(CandidateGraphError, match="sum to one"):
        augment_candidate_graph_for_marginals(graph, source * 0.9, target)
    with pytest.raises(CandidateGraphError, match="below ANN"):
        augment_candidate_graph_for_marginals(graph, source, target, evidence=1e-5)


def _complete_graph(target_size, source_size):
    return CandidateGraph(
        source_size,
        target_size,
        tuple(
            CandidateEdge(column, row, EdgeSource.BYTE_SPAN, row + column + 1.0)
            for column in range(source_size)
            for row in range(target_size)
        ),
    )


@pytest.mark.parametrize("shape", [(2, 3), (3, 2)])
def test_sparse_matches_dense_oracle_on_non_square_graph(shape):
    graph = _complete_graph(*shape)
    source = np.full(shape[1], 1 / shape[1])
    target = np.full(shape[0], 1 / shape[0])
    costs = candidate_edge_costs(graph.edges)
    dense_cost = np.full(shape, np.inf)
    for edge, cost in zip(graph.edges, costs):
        dense_cost[edge.target_id, edge.source_id] = cost
    dense, _ = dense_sinkhorn(dense_cost, source, target, epsilon=0.7, tolerance=1e-10)
    sparse, report = sparse_log_sinkhorn(
        graph, source, target, epsilon=0.7, tolerance=1e-10
    )
    np.testing.assert_allclose(sparse.to_dense(), dense, atol=1e-10)
    assert report.converged
    transport = sparse_conditional_from_coupling(sparse, source).to_dense()
    np.testing.assert_allclose(transport.sum(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(transport @ source, target, atol=1e-10)


def test_sparse_small_epsilon_is_finite_and_graph_outside_is_zero():
    graph = CandidateGraph(
        2,
        2,
        (
            CandidateEdge(0, 0, EdgeSource.EXACT_BYTE, 1.0),
            CandidateEdge(1, 1, EdgeSource.EXACT_BYTE, 1.0),
        ),
    )
    sparse, _ = sparse_log_sinkhorn(
        graph,
        np.array([1e-9, 1 - 1e-9]),
        np.array([1e-9, 1 - 1e-9]),
        epsilon=1e-9,
    )
    dense = sparse.to_dense()
    assert np.all(np.isfinite(dense))
    assert dense[0, 1] == dense[1, 0] == 0


def test_sparse_rejects_infeasible_component_and_duplicate_edges():
    disconnected = CandidateGraph(
        2,
        2,
        (
            CandidateEdge(0, 0, EdgeSource.EXACT_BYTE, 1.0),
            CandidateEdge(1, 1, EdgeSource.EXACT_BYTE, 1.0),
        ),
    )
    with pytest.raises(SinkhornError, match="infeasible component"):
        sparse_log_sinkhorn(
            disconnected,
            np.array([0.8, 0.2]),
            np.array([0.5, 0.5]),
            epsilon=1.0,
        )
    duplicate = CandidateGraph(
        1,
        1,
        (
            CandidateEdge(0, 0, EdgeSource.EXACT_BYTE, 1.0),
            CandidateEdge(0, 0, EdgeSource.EXACT_BYTE, 2.0),
        ),
    )
    with pytest.raises(SinkhornError, match="duplicate"):
        sparse_log_sinkhorn(duplicate, np.array([1.0]), np.array([1.0]), epsilon=1.0)


def test_sparse_max_iter_failure_is_explicit():
    graph = _complete_graph(2, 2)
    with pytest.raises(SinkhornError, match="did not converge"):
        sparse_log_sinkhorn(
            graph,
            np.array([0.9, 0.1]),
            np.array([0.2, 0.8]),
            epsilon=0.01,
            tolerance=1e-15,
            max_iter=1,
        )
