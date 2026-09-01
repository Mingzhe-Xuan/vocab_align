"""Training-free vocabulary transport utilities."""

from .artifact import (
    ArtifactError,
    TransportArtifact,
    artifact_from_dense,
    load_transport_artifact,
    save_transport_artifact,
)
from .sinkhorn import (
    ConvergenceReport,
    SinkhornError,
    conditional_from_coupling,
    dense_sinkhorn,
)
from .vocab_transport import (
    LocalTransportArtifact,
    build_small_transport,
    load_transport,
    save_transport,
)

__all__ = [
    "ArtifactError",
    "ConvergenceReport",
    "LocalTransportArtifact",
    "SinkhornError",
    "TransportArtifact",
    "artifact_from_dense",
    "build_small_transport",
    "conditional_from_coupling",
    "dense_sinkhorn",
    "load_transport",
    "load_transport_artifact",
    "save_transport",
    "save_transport_artifact",
]
