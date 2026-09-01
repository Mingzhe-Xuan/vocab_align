# Vocabulary transport

This package contains reusable, training-free vocabulary transport logic. It
does not modify the existing C2C projector or wrapper.

## Modules and interfaces

- `vocab_transport.py`: local special/exact/span baseline for tiny corpora.
- `sinkhorn.py`: dense oracle and sparse log-domain scaling with gauge-fixed
  convex-dual L-BFGS acceleration for ill-conditioned supports. Costs and
  couplings always have shape `[target_vocab, source_vocab]`; source and target
  marginals have shapes `[source_vocab]` and `[target_vocab]` respectively.
- `artifact.py`: validated, versioned sparse CSC serialization for a transport
  matrix and its provenance. Loading never enables NumPy pickle payloads.
- `config.py`: immutable model/data/runtime configuration with pinned revision
  and cross-field validation.
- `manifest.py`: order-independent train/dev splits based on stable sample IDs.
- `corpus.py`: pinned raw-corpus hashing, canonical conversation identities,
  exact-content deduplication, and manifest-bound split loading.
- `token_metadata.py`: raw token bytes, character-to-byte offsets, tokenizer
  fingerprints, and special/control classification shared by builders/audits.
- `baseline.py`: deterministic baseline snapshots that keep canonical messages
  separate from each model's rendered prompt and mark unavailable resources.
- `candidate_graph.py`: prioritized special/exact/span/ANN sparse support,
  followed by marginal-aware low-evidence feasibility edges that guarantee a
  strictly positive sparse coupling support rather than mere connectivity.
- `ann_candidates.py`: deterministic bidirectional byte-ngram LSH candidates,
  including low-evidence anchor bridges that connect the ordinary-token graph.
- `marginals.py`: canonical-content frequency marginals, smoothing, and active
  support filtering under explicit allowed-ID and special-token rules.
- `audit.py`: independent invariant, cost, entropy, candidate-source, and
  dangerous-special checks with JSON/Markdown rendering.
- `soft_transport.py`: temperature softmax, exact sparse transport, source
  top-m accounting, and receiver embedding expectation.
- `metrics.py`: evaluator-compatible segmented latency, length, and memory data.
- `wrapper.py`: no-grad source prefill, explicit causal-shift virtual prompts,
  receiver KV-cache decoding, and an independent receiver-only baseline path.

Generated artifacts belong under `local/transport/artifacts/` (or an explicit
runtime output directory), not in this source package.
