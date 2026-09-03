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
  status, valid-artifact resume, external ANN candidates, manifest-bound raw
  conversation mode, direct preview texts, and offline toy mode. Formal runs
  require `--records-jsonl`, `--manifest-json`, and `--build-split` together.
- `smoke_stt.py`: one pinned prompt run through receiver-only and STT with
  fingerprint-checked transport, atomic JSON shapes, quality statistics,
  segmented metrics, runtime metadata, and receiver text. Smoke generation is
  deliberately limited to one or two new tokens.
- `build_ann_candidates.py`: deterministic bidirectional byte-ngram LSH
  candidates with explicit low-evidence connectivity bridges and provenance.
- `summarize_transport.py`: atomic accuracy, subject, failure-reason, and
  numeric-metric aggregation from versioned per-sample JSONL records.
- `run_transport_ablation.py`: deterministically expand a pre-registered dev
  search or its single frozen benchmark-test configuration into atomic JSON.

Run commands as modules from the `C2C` root, for example:

```bash
python -m script.transport.freeze_baseline --help
```

The real-model functional smoke uses the accepted 500k artifact and must run
under Slurm:

```bash
sbatch script/transport/slurm/smoke_real_models.sbatch
```

The pinned five-question MMLU-Redux transport evaluation likewise runs under
Slurm and writes resumable per-sample records plus an atomic summary:

```bash
sbatch script/transport/slurm/evaluate_stt_mmlu_redux.sbatch
```

It requests one GPU without pinning a partition, validates the locked runtime
before loading model weights, refuses to overwrite an existing report, and
defaults to offline Hugging Face cache access. Its timings are diagnostics, not
benchmark results.

The exact-STT cross-benchmark smoke recipes cover GSM8K, MATH-500, and one
LongBench Qasper sample in addition to MMLU-Redux. Submit each recipe through
the generic Slurm entry point, for example:

```bash
sbatch script/transport/slurm/evaluate_stt_benchmark.sbatch \
  recipe/eval_recipe/stt_gsm8k.yaml
```

These are small functional/initial-quality samples, not full benchmark scores.
LongBench keeps the official Qasper prompt and reference fields but marks rows
as requiring the external official scorer.
