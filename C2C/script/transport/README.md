# Transport scripts

These commands parse arguments, call reusable functions from
`rosetta.transport`, and write versioned outputs. Core algorithms do not live
in this folder.

- `compare_tokenizers.py`: full-vocabulary byte/special-token audit.
- `build_small_vocab_transport.py`: small local exact/span prototype.
- `freeze_baseline.py`: freeze canonical messages, separately rendered source
  and target prompts, runtime metadata, and checkpoint availability.
- `audit_vocab_transport.py`: independently reload and audit an artifact into
  machine-readable JSON and a compact Markdown report.
- `build_vocab_transport.py`: atomic formal artifact builder with checkpoint
  status, valid-artifact resume, external ANN candidates, and offline toy mode.
- `smoke_stt.py`: one pinned STT prompt with fingerprint-checked transport,
  atomic JSON shapes, quality statistics, segmented metrics, and receiver text.
- `build_ann_candidates.py`: deterministic bidirectional byte-ngram LSH
  candidates with explicit low-evidence connectivity bridges and provenance.

Run commands as modules from the `C2C` root, for example:

```bash
python -m script.transport.freeze_baseline --help
```
