"""Versioned, safe serialization for sparse vocabulary transport matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


SCHEMA_VERSION = 1


class ArtifactError(ValueError):
    """Raised when a transport artifact is invalid or incompatible."""


@dataclass(frozen=True)
class TransportArtifact:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]
    source_marginal: np.ndarray
    target_marginal: np.ndarray
    metadata: Dict[str, Any]

    def validate(self, tolerance: float = 1e-8) -> None:
        dtype_tolerance = (
            float(np.finfo(self.data.dtype).eps) * 10
            if self.data.dtype.kind == "f"
            else tolerance
        )
        effective_tolerance = max(tolerance, dtype_tolerance)
        target_size, source_size = self.shape
        if target_size <= 0 or source_size <= 0:
            raise ArtifactError("artifact shape must be positive")
        if self.indptr.shape != (source_size + 1,):
            raise ArtifactError("CSC indptr length does not match source vocabulary")
        if self.indptr.dtype.kind not in "iu" or self.indices.dtype.kind not in "iu":
            raise ArtifactError("CSC indices must use integer dtypes")
        if self.indptr[0] != 0 or np.any(np.diff(self.indptr) < 0):
            raise ArtifactError("CSC indptr must start at zero and be monotonic")
        if self.indptr[-1] != len(self.indices) or len(self.indices) != len(self.data):
            raise ArtifactError("CSC arrays have inconsistent lengths")
        if np.any(self.indices < 0) or np.any(self.indices >= target_size):
            raise ArtifactError("CSC row index is outside target vocabulary")
        if not np.all(np.isfinite(self.data)) or np.any(self.data < 0):
            raise ArtifactError("transport values must be finite and nonnegative")
        if self.source_marginal.shape != (source_size,):
            raise ArtifactError("source marginal shape mismatch")
        if self.target_marginal.shape != (target_size,):
            raise ArtifactError("target marginal shape mismatch")
        if np.any(self.source_marginal <= 0) or np.any(self.target_marginal <= 0):
            raise ArtifactError("artifact marginals must be strictly positive")
        if not np.all(np.isfinite(self.source_marginal)) or not np.all(
            np.isfinite(self.target_marginal)
        ):
            raise ArtifactError("artifact marginals must be finite")
        if not np.isclose(self.source_marginal.sum(), 1.0) or not np.isclose(
            self.target_marginal.sum(), 1.0
        ):
            raise ArtifactError("artifact marginals must each sum to one")
        required = {
            "schema_version",
            "source_fingerprint",
            "target_fingerprint",
            "build_config",
            "seed",
            "code_version",
        }
        missing = required.difference(self.metadata)
        if missing:
            raise ArtifactError(f"artifact metadata missing: {sorted(missing)}")
        if self.metadata["schema_version"] != SCHEMA_VERSION:
            raise ArtifactError("unsupported artifact schema version")

        column_sums = np.add.reduceat(
            np.append(self.data, 0.0), self.indptr[:-1]
        )
        empty = self.indptr[1:] == self.indptr[:-1]
        column_sums[empty] = 0.0
        if np.max(np.abs(column_sums - 1.0)) > effective_tolerance:
            raise ArtifactError("transport columns are not normalized")
        transported = np.zeros(target_size, dtype=np.float64)
        for source_id in range(source_size):
            start, end = self.indptr[source_id : source_id + 2]
            transported[self.indices[start:end]] += (
                self.data[start:end] * self.source_marginal[source_id]
            )
        if np.abs(transported - self.target_marginal).sum() > effective_tolerance:
            raise ArtifactError("transported source marginal does not match target")


def artifact_from_dense(
    transport: np.ndarray,
    source_marginal: np.ndarray,
    target_marginal: np.ndarray,
    metadata: Mapping[str, Any],
) -> TransportArtifact:
    transport = np.asarray(transport)
    if transport.ndim != 2:
        raise ArtifactError("transport must have shape [target_vocab, source_vocab]")
    indices = []
    values = []
    indptr = [0]
    for column in range(transport.shape[1]):
        rows = np.flatnonzero(transport[:, column] != 0)
        indices.extend(rows.tolist())
        values.extend(transport[rows, column].tolist())
        indptr.append(len(indices))
    artifact = TransportArtifact(
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int64),
        data=np.asarray(values, dtype=transport.dtype),
        shape=(int(transport.shape[0]), int(transport.shape[1])),
        source_marginal=np.asarray(source_marginal),
        target_marginal=np.asarray(target_marginal),
        metadata=dict(metadata),
    )
    artifact.validate()
    return artifact


def save_transport_artifact(artifact: TransportArtifact, path: str | Path) -> None:
    artifact.validate()
    payload = json.dumps(artifact.metadata, sort_keys=True, separators=(",", ":"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indptr=artifact.indptr,
        indices=artifact.indices,
        data=artifact.data,
        shape=np.asarray(artifact.shape, dtype=np.int64),
        source_marginal=artifact.source_marginal,
        target_marginal=artifact.target_marginal,
        metadata=np.asarray(payload),
    )


def load_transport_artifact(
    path: str | Path,
    *,
    source_fingerprint: Optional[str] = None,
    target_fingerprint: Optional[str] = None,
) -> TransportArtifact:
    try:
        with np.load(Path(path), allow_pickle=False) as payload:
            required = {
                "indptr",
                "indices",
                "data",
                "shape",
                "source_marginal",
                "target_marginal",
                "metadata",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ArtifactError(f"artifact arrays missing: {sorted(missing)}")
            shape_values = payload["shape"]
            if shape_values.shape != (2,):
                raise ArtifactError("artifact shape field is invalid")
            metadata = json.loads(str(payload["metadata"].item()))
            artifact = TransportArtifact(
                indptr=payload["indptr"].copy(),
                indices=payload["indices"].copy(),
                data=payload["data"].copy(),
                shape=(int(shape_values[0]), int(shape_values[1])),
                source_marginal=payload["source_marginal"].copy(),
                target_marginal=payload["target_marginal"].copy(),
                metadata=metadata,
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(f"failed to load transport artifact: {exc}") from exc
    artifact.validate()
    if source_fingerprint is not None and (
        artifact.metadata["source_fingerprint"] != source_fingerprint
    ):
        raise ArtifactError("source tokenizer fingerprint mismatch")
    if target_fingerprint is not None and (
        artifact.metadata["target_fingerprint"] != target_fingerprint
    ):
        raise ArtifactError("target tokenizer fingerprint mismatch")
    return artifact
