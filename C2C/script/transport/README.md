# Transport scripts

These commands parse arguments, call reusable functions from
`rosetta.transport`, and write versioned outputs. Core algorithms do not live
in this folder.

- `compare_tokenizers.py`: full-vocabulary byte/special-token audit.
- `build_small_vocab_transport.py`: small local exact/span prototype.
- `freeze_baseline.py`: freeze canonical messages, separately rendered source
  and target prompts, runtime metadata, and checkpoint availability.

Run commands as modules from the `C2C` root, for example:

```bash
python -m script.transport.freeze_baseline --help
```
