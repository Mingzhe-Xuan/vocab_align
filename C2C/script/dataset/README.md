# Dataset scripts

`build_transport_manifest.py` reads stable sample IDs from JSONL and writes a
versioned deterministic `transport_train`/`transport_dev` manifest. The split
is based on SHA-256 ranking, so input row order cannot alter membership.

For OpenHermes, use `canonical-content` identity because its `id`, `idx`, and
`hash` columns can be null. This mode hashes normalized conversation content,
deduplicates exact conversations, and binds the manifest to both a pinned
dataset revision and the raw JSONL SHA-256.

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
