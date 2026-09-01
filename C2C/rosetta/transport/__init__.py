"""Training-free vocabulary transport utilities."""

from .artifact import (
    ArtifactError,
    TransportArtifact,
    artifact_from_dense,
    load_transport_artifact,
    save_transport_artifact,
)
from .config import ConfigError, DataSpec, ModelSpec, TransportConfig
from .manifest import (
    ManifestError,
    build_transport_manifest,
    save_manifest,
    serialize_manifest,
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
    "ConfigError",
    "DataSpec",
    "LocalTransportArtifact",
    "ManifestError",
    "ModelSpec",
    "SinkhornError",
    "TransportArtifact",
    "TransportConfig",
    "artifact_from_dense",
    "build_small_transport",
    "build_transport_manifest",
    "conditional_from_coupling",
    "dense_sinkhorn",
    "load_transport",
    "load_transport_artifact",
    "save_transport",
    "save_transport_artifact",
    "save_manifest",
    "serialize_manifest",
]
