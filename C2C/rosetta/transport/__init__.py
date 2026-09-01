"""Training-free vocabulary transport utilities."""

from .artifact import (
    ArtifactError,
    TransportArtifact,
    artifact_from_dense,
    load_transport_artifact,
    save_transport_artifact,
)
from .baseline import (
    BaselineError,
    BaselineSnapshot,
    collect_runtime_info,
    freeze_baseline,
    save_baseline,
)
from .candidate_graph import (
    CandidateEdge,
    CandidateGraph,
    CandidateGraphError,
    EdgeSource,
    accumulate_byte_span_counts,
    build_candidate_graph,
)
from .config import ConfigError, DataSpec, ModelSpec, TransportConfig
from .manifest import (
    ManifestError,
    build_transport_manifest,
    save_manifest,
    serialize_manifest,
)
from .marginals import MarginalError, TokenMarginal, estimate_token_marginal
from .sinkhorn import (
    ConvergenceReport,
    SinkhornError,
    conditional_from_coupling,
    dense_sinkhorn,
)
from .token_metadata import (
    TokenMetadata,
    encode_with_byte_spans,
    exact_byte_matches,
    iter_token_metadata,
    ordinary_bytes_index,
    special_id_to_kind,
    special_id_to_token,
    token_raw_bytes,
    tokenizer_fingerprint,
)
from .vocab_transport import (
    LocalTransportArtifact,
    build_small_transport,
    load_transport,
    save_transport,
)

__all__ = [
    "ArtifactError",
    "BaselineError",
    "BaselineSnapshot",
    "CandidateEdge",
    "CandidateGraph",
    "CandidateGraphError",
    "ConvergenceReport",
    "ConfigError",
    "DataSpec",
    "EdgeSource",
    "LocalTransportArtifact",
    "ManifestError",
    "MarginalError",
    "ModelSpec",
    "SinkhornError",
    "TransportArtifact",
    "TransportConfig",
    "TokenMetadata",
    "TokenMarginal",
    "accumulate_byte_span_counts",
    "artifact_from_dense",
    "build_small_transport",
    "build_transport_manifest",
    "build_candidate_graph",
    "conditional_from_coupling",
    "collect_runtime_info",
    "dense_sinkhorn",
    "encode_with_byte_spans",
    "exact_byte_matches",
    "estimate_token_marginal",
    "freeze_baseline",
    "iter_token_metadata",
    "load_transport",
    "load_transport_artifact",
    "ordinary_bytes_index",
    "save_transport",
    "save_transport_artifact",
    "save_manifest",
    "save_baseline",
    "serialize_manifest",
    "special_id_to_kind",
    "special_id_to_token",
    "token_raw_bytes",
    "tokenizer_fingerprint",
]
