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
explicitly a preview rather than the formal `transport_train` artifact.

Example:

```bash
mkdir -p local/transport/logs
sbatch script/transport/slurm/build_ann_candidates.sbatch

TEXTS_JSONL=local/transport/inputs/preview_texts.jsonl \
  sbatch script/transport/slurm/build_preview.sbatch

TEXTS_JSONL=local/transport/inputs/preview_texts.jsonl \
ANN_CANDIDATES_JSON=local/transport/artifacts/qwen3_8b_to_mistral_nemo_ann.json \
  sbatch script/transport/slurm/build_full_support_preview.sbatch
```
