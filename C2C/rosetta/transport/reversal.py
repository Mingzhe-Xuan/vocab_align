"""Bayes reversal for a sparse conditional transport artifact."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from .artifact import ArtifactError, TransportArtifact


REVERSAL_METHOD = "bayes-joint-reversal-v1"


def _swap_special_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    swapped = dict(mapping)
    if "source_id" not in swapped or "target_id" not in swapped:
        raise ArtifactError("special mapping must contain source_id and target_id")
    swapped["source_id"], swapped["target_id"] = (
        swapped["target_id"],
        swapped["source_id"],
    )
    return swapped


def _reversed_build_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    build_config = copy.deepcopy(dict(metadata.get("build_config", {})))
    support_policy = build_config.get("support_policy")
    if isinstance(support_policy, Mapping):
        parent_policy = copy.deepcopy(dict(support_policy))
        reversed_policy = {
            "source": parent_policy.get("target", "parent-target-active-support"),
            "target": parent_policy.get("source", "parent-source-active-support"),
            "derivation": "active supports inherited from Bayes-reversed parent",
        }
        if "source_special_fallback" in parent_policy:
            reversed_policy["parent_source_special_fallback"] = parent_policy[
                "source_special_fallback"
            ]
        build_config["support_policy"] = reversed_policy

    ann = build_config.get("ann")
    if isinstance(ann, Mapping) and ann.get("enabled"):
        build_config["ann"] = {
            "enabled": True,
            "kind": "coordinate-transposed-parent-candidate-graph",
            "parent": copy.deepcopy(dict(ann)),
        }
    build_config["derivation"] = REVERSAL_METHOD
    return build_config


def reverse_transport_artifact(
    artifact: TransportArtifact,
    *,
    code_version: str,
    parent_sha256: str | None = None,
) -> TransportArtifact:
    """Reverse ``P(target|source)`` through its realized joint coupling.

    A plain sparse transpose is not a conditional transport because its new
    columns do not generally sum to one. This function reconstructs the joint
    mass ``P(target|source) * p(source)`` and conditions it in the opposite
    direction. The reversed source marginal is the *realized* parent target
    marginal, preserving the parent joint even when its recorded Sinkhorn row
    residual is nonzero.

    Live tokenizer fingerprints are intentionally not recomputed here. Their
    recorded source/target values are swapped as direction provenance.
    """

    artifact.validate()
    if not str(code_version).strip():
        raise ArtifactError("reverse artifact code_version is required")
    if parent_sha256 is not None and (
        len(parent_sha256) != 64
        or any(character not in "0123456789abcdef" for character in parent_sha256)
    ):
        raise ArtifactError("parent_sha256 must be a lowercase SHA-256 digest")

    parent_target_size, parent_source_size = artifact.shape
    counts_by_parent_source = np.diff(artifact.indptr).astype(np.int64, copy=False)
    parent_columns = np.repeat(
        np.arange(parent_source_size, dtype=np.int64), counts_by_parent_source
    )
    joint_values = (
        artifact.data.astype(np.float64, copy=False)
        * artifact.source_marginal[parent_columns]
    )
    realized_parent_target = np.bincount(
        artifact.indices,
        weights=joint_values,
        minlength=parent_target_size,
    ).astype(np.float64, copy=False)
    if np.any(~np.isfinite(realized_parent_target)) or np.any(
        realized_parent_target <= 0
    ):
        raise ArtifactError(
            "cannot reverse an artifact with nonpositive realized target mass"
        )

    reverse_values = joint_values / realized_parent_target[artifact.indices]
    order = np.lexsort((parent_columns, artifact.indices))
    reverse_counts = np.bincount(artifact.indices, minlength=parent_target_size).astype(
        np.int64, copy=False
    )
    reverse_indptr = np.empty(parent_target_size + 1, dtype=np.int64)
    reverse_indptr[0] = 0
    np.cumsum(reverse_counts, out=reverse_indptr[1:])
    reverse_column_sums = np.bincount(
        artifact.indices, weights=reverse_values, minlength=parent_target_size
    )
    reverse_transported = np.bincount(
        parent_columns,
        weights=reverse_values * realized_parent_target[artifact.indices],
        minlength=parent_source_size,
    )
    reverse_column_residual = float(
        np.abs(
            reverse_column_sums * realized_parent_target - realized_parent_target
        ).sum()
    )
    reverse_row_residual = float(
        np.abs(reverse_transported - artifact.source_marginal).sum()
    )

    parent_target_residual = float(
        np.abs(realized_parent_target - artifact.target_marginal).sum()
    )
    identity = {
        "method": REVERSAL_METHOD,
        "parent_input_fingerprint": artifact.metadata.get("input_fingerprint"),
        "parent_sha256": parent_sha256,
        "parent_source_fingerprint": artifact.metadata["source_fingerprint"],
        "parent_target_fingerprint": artifact.metadata["target_fingerprint"],
    }
    input_fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        "schema_version": artifact.metadata["schema_version"],
        "source_fingerprint": artifact.metadata["target_fingerprint"],
        "target_fingerprint": artifact.metadata["source_fingerprint"],
        "input_fingerprint": input_fingerprint,
        "build_config": _reversed_build_config(artifact.metadata),
        "seed": artifact.metadata["seed"],
        "code_version": code_version,
        "convergence": {
            "converged": True,
            "method": REVERSAL_METHOD,
            "iterations": 0,
            "row_residual": reverse_row_residual,
            "column_residual": reverse_column_residual,
            "tolerance": artifact.metadata.get("build_config", {}).get(
                "tolerance", 1e-8
            ),
        },
        "dense_oracle_max_error": None,
        "coordinate_system": "active-support-target-by-source",
        "special_mappings": [
            _swap_special_mapping(mapping)
            for mapping in artifact.metadata.get("special_mappings", [])
        ],
        "derivation": {
            **identity,
            "fingerprint_validation": "not-performed",
            "parent_code_version": artifact.metadata.get("code_version"),
            "parent_convergence": copy.deepcopy(artifact.metadata.get("convergence")),
            "parent_recorded_target_vs_realized_l1": parent_target_residual,
            "joint_mass_preserved": True,
        },
    }
    reversed_artifact = TransportArtifact(
        indptr=reverse_indptr,
        indices=parent_columns[order],
        data=reverse_values[order],
        shape=(parent_source_size, parent_target_size),
        source_marginal=realized_parent_target,
        target_marginal=artifact.source_marginal.copy(),
        metadata=metadata,
        source_token_ids=artifact.target_token_ids.copy(),
        target_token_ids=artifact.source_token_ids.copy(),
        candidate_rows=artifact.candidate_columns.copy(),
        candidate_columns=artifact.candidate_rows.copy(),
        candidate_evidence=artifact.candidate_evidence.copy(),
        candidate_sources=artifact.candidate_sources.copy(),
    )
    reversed_artifact.validate()
    return reversed_artifact
