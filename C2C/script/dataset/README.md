# Dataset scripts

`build_transport_manifest.py` reads stable sample IDs from JSONL and writes a
versioned deterministic `transport_train`/`transport_dev` manifest. The split
is based on SHA-256 ranking, so input row order cannot alter membership.

For OpenHermes, use `canonical-content` identity because its `id`, `idx`, and
`hash` columns can be null. This mode hashes normalized conversation content,
deduplicates exact conversations, and binds the manifest to both a pinned
dataset revision and the raw JSONL SHA-256.

`materialize_transport_corpus.py` is the formal 500k entry point. It streams
the first 500,000 rows of the pinned source split, exactly matching the
`OpenHermesChatDataset` `num_samples` stage, then uses seed 42 for the stable
99/1 ID split. It writes records atomically and attaches prefix-selection
provenance to the raw-hash-bound manifest. Adapter token-length filtering is
intentionally not applied during base-corpus materialization and is recorded
as such. Without `--input-jsonl`, the Hugging Face
`datasets` package is imported lazily and the pinned dataset revision is loaded;
`--input-jsonl` exists for offline verification and pre-downloaded sources.

```bash
python -m script.dataset.materialize_transport_corpus \
  --dataset teknium/OpenHermes-2.5 \
  --dataset-revision 05c3557e57b6dd1d0e0cb8369ba53b43e15fd10b \
  --records-output local/transport/corpora/openhermes-500k.jsonl \
  --manifest-output local/transport/manifests/openhermes-500k.json \
  --cache-dir local/transport/datasets/huggingface \
  --sample-count 500000 --seed 42 --dev-fraction 0.01
```

```bash
python -m script.dataset.build_transport_manifest \
  --input local/openhermes.jsonl \
  --output local/transport/manifests/openhermes.json \
  --id-field id --seed 42 --dev-fraction 0.01
```

```bash
python -m script.dataset.build_transport_manifest \
  --input local/transport/corpora/openhermes-train.jsonl \
  --output local/transport/manifests/openhermes.json \
  --identity-mode canonical-content \
  --dataset teknium/OpenHermes-2.5 \
  --dataset-revision 05c3557e57b6dd1d0e0cb8369ba53b43e15fd10b \
  --seed 42 --dev-fraction 0.01
```
