"""Numerically stable dense Sinkhorn oracle for vocabulary transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from .candidate_graph import CandidateEdge, CandidateGraph


class SinkhornError(ValueError):
    """Raised when the transport problem is invalid or fails to converge."""


@dataclass(frozen=True)
class ConvergenceReport:
    iterations: int
    row_residual: float
    column_residual: float
    converged: bool
    tolerance: float
    max_iter: int
    method: str = "sinkhorn"
    sinkhorn_iterations: int | None = None
    acceleration_evaluations: int = 0
    acceleration_attempts: int = 0
    acceleration_terminations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SparseCoupling:
    row_indices: np.ndarray
    column_indices: np.ndarray
    data: np.ndarray
    shape: Tuple[int, int]

    def to_dense(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=self.data.dtype)
        dense[self.row_indices, self.column_indices] = self.data
        return dense


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite_maximum = np.isfinite(maximum)
    shifted = np.where(finite_maximum, values - maximum, -np.inf)
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = np.where(finite_maximum, maximum + np.log(total), -np.inf)
    return np.squeeze(result, axis=axis)


def _validate_problem(
    cost: np.ndarray, source_marginal: np.ndarray, target_marginal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cost = np.asarray(cost, dtype=np.float64)
    source = np.asarray(source_marginal, dtype=np.float64)
    target = np.asarray(target_marginal, dtype=np.float64)
    if cost.ndim != 2:
        raise SinkhornError("cost must have shape [target_vocab, source_vocab]")
    if source.shape != (cost.shape[1],) or target.shape != (cost.shape[0],):
        raise SinkhornError(
            "marginal shapes must match cost [target_vocab, source_vocab]"
        )
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise SinkhornError("marginals must be finite")
    if np.any(source <= 0) or np.any(target <= 0):
        raise SinkhornError("active-support marginals must be strictly positive")
    if not np.isclose(source.sum(), 1.0) or not np.isclose(target.sum(), 1.0):
        raise SinkhornError("source and target marginals must each sum to one")
    if np.any(np.isnan(cost)) or np.any(np.isneginf(cost)):
        raise SinkhornError("cost may contain +inf for missing edges, but not NaN/-inf")
    support = np.isfinite(cost)
    if not np.all(support.any(axis=0)):
        raise SinkhornError("each positive-mass source token needs a candidate edge")
    if not np.all(support.any(axis=1)):
        raise SinkhornError("each positive-mass target token needs a candidate edge")
    return cost, source, target


def dense_sinkhorn(
    cost: np.ndarray,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    *,
    epsilon: float,
    tolerance: float = 1e-9,
    max_iter: int = 10_000,
) -> Tuple[np.ndarray, ConvergenceReport]:
    """Return coupling ``Pi[target, source]`` and its convergence report.

    The implementation operates in the log domain, so small positive epsilon
    values and large finite costs do not silently underflow kernel entries.
    Missing candidate edges are represented by ``+inf`` and retain zero mass.
    """
    cost, source, target = _validate_problem(cost, source_marginal, target_marginal)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise SinkhornError("epsilon must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise SinkhornError("tolerance must be finite and positive")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise SinkhornError("max_iter must be a positive integer")

    log_kernel = -cost / epsilon
    log_source = np.log(source)
    log_target = np.log(target)
    log_v = np.zeros_like(source)
    coupling = np.empty_like(cost)
    row_residual = column_residual = float("inf")

    for iteration in range(1, max_iter + 1):
        row_norm = _logsumexp(log_kernel + log_v[None, :], axis=1)
        if not np.all(np.isfinite(row_norm)):
            raise SinkhornError("candidate support is infeasible for target marginal")
        log_u = log_target - row_norm
        column_norm = _logsumexp(log_kernel + log_u[:, None], axis=0)
        if not np.all(np.isfinite(column_norm)):
            raise SinkhornError("candidate support is infeasible for source marginal")
        log_v = log_source - column_norm

        log_coupling = log_u[:, None] + log_kernel + log_v[None, :]
        coupling = np.exp(log_coupling)
        if not np.all(np.isfinite(coupling)):
            raise SinkhornError("Sinkhorn produced a non-finite coupling")
        row_residual = float(np.abs(coupling.sum(axis=1) - target).sum())
        column_residual = float(np.abs(coupling.sum(axis=0) - source).sum())
        if max(row_residual, column_residual) <= tolerance:
            return coupling, ConvergenceReport(
                iterations=iteration,
                row_residual=row_residual,
                column_residual=column_residual,
                converged=True,
                tolerance=tolerance,
                max_iter=max_iter,
            )

    report = ConvergenceReport(
        iterations=max_iter,
        row_residual=row_residual,
        column_residual=column_residual,
        converged=False,
        tolerance=tolerance,
        max_iter=max_iter,
    )
    raise SinkhornError(f"Sinkhorn did not converge: {report.to_dict()}")


def conditional_from_coupling(
    coupling: np.ndarray, source_marginal: np.ndarray
) -> np.ndarray:
    """Convert ``Pi`` to column-stochastic ``T = Pi Diag(a)^-1``."""
    coupling = np.asarray(coupling)
    source = np.asarray(source_marginal)
    if coupling.ndim != 2 or source.shape != (coupling.shape[1],):
        raise SinkhornError("source marginal must match coupling source dimension")
    if not np.all(np.isfinite(coupling)) or np.any(coupling < 0):
        raise SinkhornError("coupling must be finite and nonnegative")
    if not np.all(np.isfinite(source)) or np.any(source <= 0):
        raise SinkhornError("source marginal must be finite and strictly positive")
    return coupling / source[None, :]


def candidate_edge_costs(
    edges: Iterable[CandidateEdge], *, delta: float = 1e-12
) -> np.ndarray:
    """Turn positive edge evidence into per-source normalized negative logs."""
    edges = tuple(edges)
    if not np.isfinite(delta) or delta <= 0:
        raise SinkhornError("delta must be finite and positive")
    if any(not np.isfinite(edge.evidence) or edge.evidence <= 0 for edge in edges):
        raise SinkhornError("candidate evidence must be finite and positive")
    totals: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for edge in edges:
        totals[edge.source_id] = totals.get(edge.source_id, 0.0) + edge.evidence
        counts[edge.source_id] = counts.get(edge.source_id, 0) + 1
    return np.asarray(
        [
            -np.log(
                (edge.evidence + delta)
                / (totals[edge.source_id] + delta * counts[edge.source_id])
            )
            for edge in edges
        ],
        dtype=np.float64,
    )


def _validate_sparse_feasibility(
    rows: np.ndarray,
    columns: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    tolerance: float,
) -> None:
    active_rows = set(np.flatnonzero(target > 0).tolist())
    active_columns = set(np.flatnonzero(source > 0).tolist())
    if active_rows.difference(rows.tolist()):
        raise SinkhornError("each positive-mass target token needs a candidate edge")
    if active_columns.difference(columns.tolist()):
        raise SinkhornError("each positive-mass source token needs a candidate edge")

    row_neighbors: Dict[int, set[int]] = {row: set() for row in active_rows}
    column_neighbors: Dict[int, set[int]] = {column: set() for column in active_columns}
    for row, column in zip(rows.tolist(), columns.tolist()):
        row_neighbors[row].add(column)
        column_neighbors[column].add(row)
    unseen_rows = set(active_rows)
    while unseen_rows:
        component_rows = {unseen_rows.pop()}
        component_columns: set[int] = set()
        frontier_rows = list(component_rows)
        while frontier_rows:
            row = frontier_rows.pop()
            for column in row_neighbors[row].difference(component_columns):
                component_columns.add(column)
                for neighbor_row in column_neighbors[column].difference(component_rows):
                    component_rows.add(neighbor_row)
                    unseen_rows.discard(neighbor_row)
                    frontier_rows.append(neighbor_row)
        row_mass = float(target[list(component_rows)].sum())
        column_mass = float(source[list(component_columns)].sum())
        if abs(row_mass - column_mass) > tolerance:
            raise SinkhornError("candidate support has a marginal-infeasible component")


def _dual_value_gradient(
    variables: np.ndarray,
    row_positions: np.ndarray,
    column_positions: np.ndarray,
    log_kernel: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the gauge-fixed entropic OT scaling dual and gradient."""
    row_count = len(target)
    column_count = len(source)
    if variables.shape != (row_count + column_count - 1,):
        raise SinkhornError("dual variable shape does not match active marginals")
    log_u = variables[:row_count]
    log_v = np.concatenate((variables[row_count:], np.zeros(1, dtype=np.float64)))
    log_mass = log_kernel + log_u[row_positions] + log_v[column_positions]
    if np.max(log_mass, initial=-np.inf) > 700:
        return float("inf"), np.full_like(variables, 1e100), np.empty(0)
    mass = np.exp(log_mass)
    if not np.all(np.isfinite(mass)):
        return float("inf"), np.full_like(variables, 1e100), np.empty(0)
    row_mass = np.bincount(row_positions, weights=mass, minlength=row_count)
    column_mass = np.bincount(column_positions, weights=mass, minlength=column_count)
    value = float(mass.sum() - np.dot(target, log_u) - np.dot(source, log_v))
    gradient = np.concatenate((row_mass - target, (column_mass - source)[:-1]))
    return value, gradient, mass


def _scaled_dual_hessian_product(
    vector: np.ndarray,
    row_positions: np.ndarray,
    column_positions: np.ndarray,
    mass: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Apply the gauge-fixed dual Hessian in sqrt-marginal coordinates."""
    row_count = len(target)
    column_count = len(source)
    expected_shape = (row_count + column_count - 1,)
    if vector.shape != expected_shape or mass.shape != row_positions.shape:
        raise SinkhornError("dual Hessian input shape does not match active support")
    variable_scale = np.sqrt(np.concatenate((target, source[:-1])))
    unscaled = vector / variable_scale
    row_direction = unscaled[:row_count]
    column_direction = np.concatenate((unscaled[row_count:], np.zeros(1)))
    edge_direction = row_direction[row_positions] + column_direction[column_positions]
    weighted = mass * edge_direction
    row_product = np.bincount(row_positions, weights=weighted, minlength=row_count)
    column_product = np.bincount(
        column_positions, weights=weighted, minlength=column_count
    )
    return np.concatenate((row_product, column_product[:-1])) / variable_scale


class _EvaluationBudgetExceeded(RuntimeError):
    pass


def _accelerate_sparse_dual(
    rows: np.ndarray,
    columns: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    active_rows: np.ndarray,
    active_columns: np.ndarray,
    log_u: np.ndarray,
    log_v: np.ndarray,
    current_data: np.ndarray,
    current_row_residual: float,
    current_column_residual: float,
    *,
    max_evaluations: int,
    cg_max_iterations: int,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, str]:
    from scipy.sparse.linalg import LinearOperator, cg

    row_lookup = np.full(len(target), -1, dtype=np.int64)
    column_lookup = np.full(len(source), -1, dtype=np.int64)
    row_lookup[active_rows] = np.arange(len(active_rows))
    column_lookup[active_columns] = np.arange(len(active_columns))
    row_positions = row_lookup[rows]
    column_positions = column_lookup[columns]
    gauge = float(log_v[active_columns[-1]])
    initial = np.concatenate(
        (
            log_u[active_rows] + gauge,
            (log_v[active_columns] - gauge)[:-1],
        )
    )
    active_source = source[active_columns]
    active_target = target[active_rows]
    variable_scale = np.sqrt(np.concatenate((active_target, active_source[:-1])))
    current_mass = current_data.copy()
    row_count = len(active_rows)
    column_count = len(active_columns)
    evaluation_count = 0
    current_metric = max(current_row_residual, current_column_residual)
    terminations: list[str] = []
    newton_steps = 0

    while evaluation_count + 2 <= max_evaluations and current_metric > tolerance:
        row_mass = np.bincount(row_positions, weights=current_mass, minlength=row_count)
        column_mass = np.bincount(
            column_positions, weights=current_mass, minlength=column_count
        )
        gradient = np.concatenate(
            ((row_mass - active_target), (column_mass - active_source)[:-1])
        )
        scaled_gradient = gradient / variable_scale
        diagonal = np.concatenate((row_mass, column_mass[:-1])) / variable_scale**2
        if (
            not np.all(np.isfinite(scaled_gradient))
            or not np.all(np.isfinite(diagonal))
            or np.any(diagonal <= 0)
        ):
            terminations.append("nonfinite-gradient-or-preconditioner")
            break

        last_direction: np.ndarray | None = None

        def hessian_product(vector: np.ndarray) -> np.ndarray:
            nonlocal evaluation_count
            if evaluation_count >= max_evaluations - 1:
                raise _EvaluationBudgetExceeded
            evaluation_count += 1
            return _scaled_dual_hessian_product(
                vector,
                row_positions,
                column_positions,
                current_mass,
                active_source,
                active_target,
            )

        def remember_direction(vector: np.ndarray) -> None:
            nonlocal last_direction
            last_direction = vector.copy()

        operator = LinearOperator(
            (len(initial), len(initial)), matvec=hessian_product, dtype=np.float64
        )
        preconditioner = LinearOperator(
            (len(initial), len(initial)),
            matvec=lambda vector: vector / diagonal,
            dtype=np.float64,
        )
        cg_iterations = min(cg_max_iterations, max_evaluations - evaluation_count - 1)
        if cg_iterations <= 0:
            break
        try:
            direction, info = cg(
                operator,
                -scaled_gradient,
                rtol=1e-6,
                atol=0.0,
                maxiter=cg_iterations,
                M=preconditioner,
                callback=remember_direction,
            )
            cg_status = f"cg-info={info}"
        except _EvaluationBudgetExceeded:
            if last_direction is None:
                terminations.append("budget-before-cg-direction")
                break
            direction = last_direction
            cg_status = "cg-budget"
        if not np.all(np.isfinite(direction)):
            terminations.append(f"{cg_status}:nonfinite-direction")
            break

        unscaled_direction = direction / variable_scale
        row_direction = unscaled_direction[:row_count]
        column_direction = np.concatenate((unscaled_direction[row_count:], np.zeros(1)))
        edge_direction = (
            row_direction[row_positions] + column_direction[column_positions]
        )
        accepted = False
        accepted_step = 0.0
        for exponent in range(12):
            if evaluation_count >= max_evaluations:
                break
            step = 0.5**exponent
            scaled_edge_direction = step * edge_direction
            evaluation_count += 1
            if np.max(scaled_edge_direction, initial=-np.inf) > 700:
                continue
            candidate_mass = current_mass * np.exp(scaled_edge_direction)
            if not np.all(np.isfinite(candidate_mass)):
                continue
            candidate_row_mass = np.bincount(
                row_positions, weights=candidate_mass, minlength=row_count
            )
            candidate_column_mass = np.bincount(
                column_positions, weights=candidate_mass, minlength=column_count
            )
            row_residual = float(np.abs(candidate_row_mass - active_target).sum())
            column_residual = float(np.abs(candidate_column_mass - active_source).sum())
            candidate_metric = max(row_residual, column_residual)
            if candidate_metric < current_metric:
                initial += step * unscaled_direction
                current_mass = candidate_mass
                current_row_residual = row_residual
                current_column_residual = column_residual
                current_metric = candidate_metric
                accepted = True
                accepted_step = step
                newton_steps += 1
                break
        terminations.append(
            f"{cg_status}:step={accepted_step:.12g}:residual={current_metric:.12g}"
        )
        if not accepted:
            break

    accelerated_u = log_u.copy()
    accelerated_v = log_v.copy()
    accelerated_u[active_rows] = initial[: len(active_rows)]
    active_v = np.concatenate(
        (initial[len(active_rows) :], np.zeros(1, dtype=np.float64))
    )
    accelerated_v[active_columns] = active_v
    termination = f"newton-steps={newton_steps};" + "|".join(terminations)
    return (
        accelerated_u,
        accelerated_v,
        current_mass,
        current_row_residual,
        current_column_residual,
        evaluation_count,
        termination,
    )


def sparse_log_sinkhorn(
    graph: CandidateGraph,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    *,
    epsilon: float,
    tolerance: float = 1e-9,
    max_iter: int = 10_000,
    delta: float = 1e-12,
    acceleration_after: int | None = 250,
    acceleration_max_evaluations: int = 1_000,
    acceleration_cg_iterations: int = 32,
) -> Tuple[SparseCoupling, ConvergenceReport]:
    """Run log-domain Sinkhorn directly on candidate edges.

    Zero-mass vocabulary entries are removed from the active OT support before
    iteration. The returned sparse coordinates retain full vocabulary IDs.
    """
    source = np.asarray(source_marginal, dtype=np.float64)
    target = np.asarray(target_marginal, dtype=np.float64)
    if source.shape != (graph.source_vocab_size,) or target.shape != (
        graph.target_vocab_size,
    ):
        raise SinkhornError("marginal shapes must match candidate graph vocabularies")
    if (
        not np.all(np.isfinite(source))
        or not np.all(np.isfinite(target))
        or np.any(source < 0)
        or np.any(target < 0)
    ):
        raise SinkhornError("sparse marginals must be finite and nonnegative")
    if not np.isclose(source.sum(), 1.0) or not np.isclose(target.sum(), 1.0):
        raise SinkhornError("source and target marginals must each sum to one")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise SinkhornError("epsilon must be finite and positive")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise SinkhornError("tolerance must be finite and positive")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise SinkhornError("max_iter must be a positive integer")
    if acceleration_after is not None and (
        not isinstance(acceleration_after, int) or acceleration_after <= 0
    ):
        raise SinkhornError("acceleration_after must be a positive integer or None")
    if (
        not isinstance(acceleration_max_evaluations, int)
        or acceleration_max_evaluations <= 0
    ):
        raise SinkhornError("acceleration evaluation budget must be positive")
    if (
        not isinstance(acceleration_cg_iterations, int)
        or acceleration_cg_iterations <= 0
    ):
        raise SinkhornError("acceleration CG iterations must be positive")

    active_edges = tuple(
        edge
        for edge in graph.edges
        if source[edge.source_id] > 0 and target[edge.target_id] > 0
    )
    pairs = [(edge.target_id, edge.source_id) for edge in active_edges]
    if len(pairs) != len(set(pairs)):
        raise SinkhornError("candidate graph contains duplicate row/column edges")
    rows = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    columns = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    _validate_sparse_feasibility(rows, columns, source, target, tolerance)
    costs = candidate_edge_costs(active_edges, delta=delta)
    log_kernel = -costs / epsilon
    active_rows = np.flatnonzero(target > 0)
    active_columns = np.flatnonzero(source > 0)
    log_source = np.full_like(source, -np.inf)
    log_target = np.full_like(target, -np.inf)
    log_source[active_columns] = np.log(source[active_columns])
    log_target[active_rows] = np.log(target[active_rows])
    log_u = np.full_like(target, -np.inf)
    log_v = np.full_like(source, -np.inf)
    log_v[active_columns] = 0.0
    row_residual = column_residual = float("inf")
    acceleration_evaluations = 0
    acceleration_attempts = 0
    acceleration_terminations: list[str] = []
    next_acceleration = acceleration_after

    for iteration in range(1, max_iter + 1):
        row_norm = np.full_like(target, -np.inf)
        np.logaddexp.at(row_norm, rows, log_kernel + log_v[columns])
        if not np.all(np.isfinite(row_norm[active_rows])):
            raise SinkhornError("candidate support is infeasible for target marginal")
        log_u[active_rows] = log_target[active_rows] - row_norm[active_rows]
        column_norm = np.full_like(source, -np.inf)
        np.logaddexp.at(column_norm, columns, log_kernel + log_u[rows])
        if not np.all(np.isfinite(column_norm[active_columns])):
            raise SinkhornError("candidate support is infeasible for source marginal")
        log_v[active_columns] = log_source[active_columns] - column_norm[active_columns]
        data = np.exp(log_u[rows] + log_kernel + log_v[columns])
        if not np.all(np.isfinite(data)):
            raise SinkhornError("sparse Sinkhorn produced non-finite coupling")
        row_mass = np.zeros_like(target)
        column_mass = np.zeros_like(source)
        np.add.at(row_mass, rows, data)
        np.add.at(column_mass, columns, data)
        row_residual = float(np.abs(row_mass - target).sum())
        column_residual = float(np.abs(column_mass - source).sum())
        if max(row_residual, column_residual) <= tolerance:
            coupling = SparseCoupling(rows, columns, data, (len(target), len(source)))
            return coupling, ConvergenceReport(
                iteration + acceleration_evaluations,
                row_residual,
                column_residual,
                True,
                tolerance,
                max_iter,
                (
                    "sinkhorn-scaled-newton-cg-sinkhorn"
                    if acceleration_attempts
                    else "sinkhorn"
                ),
                iteration,
                acceleration_evaluations,
                acceleration_attempts,
                tuple(acceleration_terminations),
            )
        if (
            next_acceleration is not None
            and iteration >= next_acceleration
            and acceleration_evaluations < acceleration_max_evaluations
            and iteration + acceleration_evaluations < max_iter
        ):
            budget = min(
                acceleration_max_evaluations - acceleration_evaluations,
                max_iter - iteration - acceleration_evaluations,
            )
            (
                log_u,
                log_v,
                data,
                row_residual,
                column_residual,
                evaluations,
                termination,
            ) = _accelerate_sparse_dual(
                rows,
                columns,
                source,
                target,
                active_rows,
                active_columns,
                log_u,
                log_v,
                data,
                row_residual,
                column_residual,
                max_evaluations=budget,
                cg_max_iterations=acceleration_cg_iterations,
                tolerance=tolerance,
            )
            acceleration_evaluations += evaluations
            acceleration_attempts += 1
            acceleration_terminations.append(termination)
            next_acceleration = iteration + acceleration_after
            if max(row_residual, column_residual) <= tolerance:
                coupling = SparseCoupling(
                    rows, columns, data, (len(target), len(source))
                )
                return coupling, ConvergenceReport(
                    iteration + acceleration_evaluations,
                    row_residual,
                    column_residual,
                    True,
                    tolerance,
                    max_iter,
                    "sinkhorn-scaled-newton-cg-sinkhorn",
                    iteration,
                    acceleration_evaluations,
                    acceleration_attempts,
                    tuple(acceleration_terminations),
                )
        if iteration + acceleration_evaluations >= max_iter:
            break
    report = ConvergenceReport(
        iteration + acceleration_evaluations,
        row_residual,
        column_residual,
        False,
        tolerance,
        max_iter,
        "sinkhorn-scaled-newton-cg-sinkhorn" if acceleration_attempts else "sinkhorn",
        iteration,
        acceleration_evaluations,
        acceleration_attempts,
        tuple(acceleration_terminations),
    )
    raise SinkhornError(f"sparse Sinkhorn did not converge: {report.to_dict()}")


def sparse_conditional_from_coupling(
    coupling: SparseCoupling, source_marginal: np.ndarray
) -> SparseCoupling:
    source = np.asarray(source_marginal)
    if source.shape != (coupling.shape[1],) or np.any(
        source[coupling.column_indices] <= 0
    ):
        raise SinkhornError("source marginal does not cover sparse coupling columns")
    return SparseCoupling(
        coupling.row_indices.copy(),
        coupling.column_indices.copy(),
        coupling.data / source[coupling.column_indices],
        coupling.shape,
    )
