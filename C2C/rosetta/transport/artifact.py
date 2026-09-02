"""Versioned, safe serialization for sparse vocabulary transport matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np


SCHEMA_VERSION = 1
MAX_MARGINAL_L1_TOLERANCE = 2e-3


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
    source_token_ids: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64)
    )
    target_token_ids: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64)
    )
    candidate_rows: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64)
    )
    candidate_columns: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.int64)
    )
    candidate_evidence: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=np.float64)
    )
    candidate_sources: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype="U16")
    )

    def validate(
        self,
        tolerance: float = 1e-8,
        *,
        marginal_tolerance: Optional[float] = None,
    ) -> None:
        dtype_tolerance = (
            float(np.finfo(self.data.dtype).eps) * 10
            if self.data.dtype.kind == "f"
            else tolerance
        )
        effective_tolerance = max(tolerance, dtype_tolerance)
        target_size, source_size = self.shape
        if target_size <= 0 or source_size <= 0:
            raise ArtifactError("artifact shape must be positive")
        source_token_ids = (
            self.source_token_ids
            if len(self.source_token_ids)
            else np.arange(source_size, dtype=np.int64)
        )
        target_token_ids = (
            self.target_token_ids
            if len(self.target_token_ids)
            else np.arange(target_size, dtype=np.int64)
        )
        if source_token_ids.shape != (source_size,) or target_token_ids.shape != (
            target_size,
        ):
            raise ArtifactError("active token ID mapping shape mismatch")
        if (
            source_token_ids.dtype.kind not in "iu"
            or target_token_ids.dtype.kind not in "iu"
            or np.any(source_token_ids < 0)
            or np.any(target_token_ids < 0)
            or len(np.unique(source_token_ids)) != source_size
            or len(np.unique(target_token_ids)) != target_size
        ):
            raise ArtifactError(
                "active token ID mappings must be unique nonnegative integers"
            )
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
        if marginal_tolerance is None:
            marginal_tolerance = self.metadata.get("build_config", {}).get(
                "tolerance", tolerance
            )
        if (
            isinstance(marginal_tolerance, bool)
            or not isinstance(marginal_tolerance, (int, float))
            or not np.isfinite(float(marginal_tolerance))
            or float(marginal_tolerance) <= 0
        ):
            raise ArtifactError(
                "artifact marginal tolerance must be finite and positive"
            )
        if float(marginal_tolerance) > MAX_MARGINAL_L1_TOLERANCE:
            raise ArtifactError(
                "artifact marginal tolerance exceeds the pre-registered maximum"
            )
        effective_marginal_tolerance = max(float(marginal_tolerance), dtype_tolerance)

        candidate_lengths = {
            len(self.candidate_rows),
            len(self.candidate_columns),
            len(self.candidate_evidence),
            len(self.candidate_sources),
        }
        if len(candidate_lengths) != 1:
            raise ArtifactError("candidate graph arrays have inconsistent lengths")
        if len(self.candidate_rows):
            if (
                self.candidate_rows.dtype.kind not in "iu"
                or self.candidate_columns.dtype.kind not in "iu"
                or np.any(self.candidate_rows < 0)
                or np.any(self.candidate_rows >= target_size)
                or np.any(self.candidate_columns < 0)
                or np.any(self.candidate_columns >= source_size)
            ):
                raise ArtifactError("candidate graph index is outside active support")
            if not np.all(np.isfinite(self.candidate_evidence)) or np.any(
                self.candidate_evidence <= 0
            ):
                raise ArtifactError("candidate evidence must be finite and positive")
            if self.candidate_sources.dtype.kind not in "US":
                raise ArtifactError("candidate source labels must be strings")

        column_sums = np.add.reduceat(np.append(self.data, 0.0), self.indptr[:-1])
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
        if (
            np.abs(transported - self.target_marginal).sum()
            > effective_marginal_tolerance
        ):
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
        source_token_ids=np.arange(transport.shape[1], dtype=np.int64),
        target_token_ids=np.arange(transport.shape[0], dtype=np.int64),
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
        source_token_ids=(
            artifact.source_token_ids
            if len(artifact.source_token_ids)
            else np.arange(artifact.shape[1], dtype=np.int64)
        ),
        target_token_ids=(
            artifact.target_token_ids
            if len(artifact.target_token_ids)
            else np.arange(artifact.shape[0], dtype=np.int64)
        ),
        candidate_rows=artifact.candidate_rows,
        candidate_columns=artifact.candidate_columns,
        candidate_evidence=artifact.candidate_evidence,
        candidate_sources=artifact.candidate_sources,
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
                source_token_ids=(
                    payload["source_token_ids"].copy()
                    if "source_token_ids" in payload.files
                    else np.arange(int(shape_values[1]), dtype=np.int64)
                ),
                target_token_ids=(
                    payload["target_token_ids"].copy()
                    if "target_token_ids" in payload.files
                    else np.arange(int(shape_values[0]), dtype=np.int64)
                ),
                candidate_rows=(
                    payload["candidate_rows"].copy()
                    if "candidate_rows" in payload.files
                    else np.asarray([], dtype=np.int64)
                ),
                candidate_columns=(
                    payload["candidate_columns"].copy()
                    if "candidate_columns" in payload.files
                    else np.asarray([], dtype=np.int64)
                ),
                candidate_evidence=(
                    payload["candidate_evidence"].copy()
                    if "candidate_evidence" in payload.files
                    else np.asarray([], dtype=np.float64)
                ),
                candidate_sources=(
                    payload["candidate_sources"].copy()
                    if "candidate_sources" in payload.files
                    else np.asarray([], dtype="U16")
                ),
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
