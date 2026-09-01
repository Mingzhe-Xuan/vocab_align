# Vocabulary transport

This package contains reusable, training-free vocabulary transport logic. It
does not modify the existing C2C projector or wrapper.

## Modules and interfaces

- `vocab_transport.py`: local special/exact/span baseline for tiny corpora.
- `sinkhorn.py`: dense log-domain Sinkhorn oracle. Costs and couplings always
  have shape `[target_vocab, source_vocab]`; source and target marginals have
  shapes `[source_vocab]` and `[target_vocab]` respectively.
- `artifact.py`: validated, versioned sparse CSC serialization for a transport
  matrix and its provenance. Loading never enables NumPy pickle payloads.
- `config.py`: immutable model/data/runtime configuration with pinned revision
  and cross-field validation.
- `manifest.py`: order-independent train/dev splits based on stable sample IDs.

Generated artifacts belong under `local/transport/artifacts/` (or an explicit
runtime output directory), not in this source package.
