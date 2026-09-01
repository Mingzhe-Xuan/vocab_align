# Dataset scripts

`build_transport_manifest.py` reads stable sample IDs from JSONL and writes a
versioned deterministic `transport_train`/`transport_dev` manifest. The split
is based on SHA-256 ranking, so input row order cannot alter membership.

```bash
python -m script.dataset.build_transport_manifest \
  --input local/openhermes.jsonl \
  --output local/transport/manifests/openhermes.json \
  --id-field id --seed 42 --dev-fraction 0.01
```
