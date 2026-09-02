# Transport Slurm jobs

Submit these jobs from the `C2C` repository root after the server worktree has
been updated with `git pull`. They never edit tracked source files; inputs,
checkpoints, logs, artifacts, and audits stay below `local/transport/`.

`build_preview.sbatch` builds a real-tokenizer, zero-smoothing preview artifact
from a canonical JSONL text file. It intentionally does not select a partition;
use the cluster default or pass an approved partition to `sbatch` after checking
`sinfo`. Optional external ANN candidates can be supplied with
`ANN_CANDIDATES_JSON`.

`build_ann_candidates.sbatch` generates the deterministic bidirectional ANN
candidate JSON used by smoothed full-support builds. Run it through Slurm
because hashing both complete vocabularies is batch processing.

`build_full_support_preview.sbatch` combines that structured ANN JSON with a
small canonical JSONL corpus and positive smoothing. It activates both complete
ordinary vocabularies to exercise sparse Sinkhorn and artifact auditing, but is
explicitly a preview rather than the formal `transport_train` artifact. Its
full-vocabulary L1 residual tolerance defaults to `2e-3`; set
`TRANSPORT_TOLERANCE` only when a different pre-registered requirement applies.
The ordinary preview job and library/CLI defaults retain their high-precision
`1e-9` behavior.

`materialize_openhermes_500k.sbatch` downloads/loads the pinned OpenHermes
revision and materializes its first 500,000 source rows plus JSONL/manifest
materialization entirely inside Slurm. It requires `datasets==4.0.0` in the
task Python venv and writes only to ignored corpus/manifest/cache/log paths.

`build_formal_transport.sbatch` consumes that hash-bound manifest's
`transport_train` split and the structured ANN candidates to build the formal
Qwen3-8B to Mistral-Nemo artifact. It defaults to the registered `2e-3`
full-vocabulary marginal tolerance, writes an independent audit/checkpoint,
and reserves 64G for up to 24 hours because canonicalizing and tokenizing the
500k corpus is batch work. `BUILD_SPLIT` exists for explicit dev validation;
the formal default remains `transport_train`.

Example:

```bash
mkdir -p local/transport/logs
sbatch script/transport/slurm/build_ann_candidates.sbatch

TEXTS_JSONL=local/transport/inputs/preview_texts.jsonl \
  sbatch script/transport/slurm/build_preview.sbatch

TEXTS_JSONL=local/transport/inputs/preview_texts.jsonl \
ANN_CANDIDATES_JSON=local/transport/artifacts/qwen3_8b_to_mistral_nemo_ann.json \
  sbatch script/transport/slurm/build_full_support_preview.sbatch

sbatch script/transport/slurm/build_formal_transport.sbatch
```
