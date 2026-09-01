"""Numerically stable dense Sinkhorn oracle for vocabulary transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    cost, source, target = _validate_problem(
        cost, source_marginal, target_marginal
    )
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
