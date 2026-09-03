# Vocabulary transport

This package contains reusable, training-free vocabulary transport logic. It
does not modify the existing C2C projector or wrapper.

## Modules and interfaces

- `ablation.py`: deterministic pre-registered dev expansion, frozen test
  configurations, and sample-ID paired accuracy summaries with explicit
  missing-pair reports.
- `vocab_transport.py`: local special/exact/span baseline for tiny corpora.
- `sinkhorn.py`: dense oracle and sparse log-domain scaling with gauge-fixed,
  marginal-scaled Newton-CG acceleration and residual backtracking for
  ill-conditioned supports. Costs and
  couplings always have shape `[target_vocab, source_vocab]`; source and target
  marginals have shapes `[source_vocab]` and `[target_vocab]` respectively.
- `artifact.py`: validated, versioned sparse CSC serialization for a transport
  matrix and its provenance. Loading never enables NumPy pickle payloads;
  column normalization stays at dtype precision while the recorded marginal
  L1 tolerance is bounded by the pre-registered full-vocabulary requirement.
- `config.py`: immutable model/data/runtime configuration with pinned revision,
  construction (`epsilon`, tolerance, iterations, smoothing), inference, and
  cross-field validation.
- `manifest.py`: order-independent train/dev splits based on stable sample IDs.
- `corpus.py`: pinned raw-corpus hashing, canonical conversation identities,
  exact-content deduplication, and manifest-bound split loading.
- `corpus_materialization.py`: streaming pinned-source-prefix materialization
  matching `OpenHermesChatDataset` sample limiting, atomic JSONL publication,
  and selection-provenance manifests.
- `approximations.py`: deterministic transport-hard embeddings, edge-chunk
  sparse accumulation, precomputed source values, and zero-safe approximation
  error metrics.
- `orf.py`: seeded block-orthogonal positive random features, sparse-transport
  vocabulary-block pre-aggregation, and online `u @ S.T / (u @ z)` mapping.
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
- `evaluation.py`: append-only per-sample records, prompt-safe resume, explicit
  failures, deterministic rank merge, and atomic aggregate summaries shared by
  receiver/source/T2T/C2C/STT adapters.
- `wrapper.py`: no-grad source prefill, explicit causal-shift virtual prompts,
  receiver KV-cache decoding, and an independent receiver-only baseline path.
  Exact soft transport partitions the source sequence into 32-token query
  chunks before applying the same full-vocabulary softmax and complete sparse
  transport. This bounds the query-by-edge intermediate without changing the
  exact transport distribution or enabling an approximation mode.

Generated artifacts belong under `local/transport/artifacts/` (or an explicit
runtime output directory), not in this source package.
