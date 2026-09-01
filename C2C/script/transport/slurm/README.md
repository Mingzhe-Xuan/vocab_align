# Transport Slurm jobs

Submit these jobs from the `C2C` repository root after the server worktree has
been updated with `git pull`. They never edit tracked source files; inputs,
checkpoints, logs, artifacts, and audits stay below `local/transport/`.

`build_preview.sbatch` builds a real-tokenizer, zero-smoothing preview artifact
from a canonical JSONL text file. It intentionally does not select a partition;
use the cluster default or pass an approved partition to `sbatch` after checking
`sinfo`. Optional external ANN candidates can be supplied with
`ANN_CANDIDATES_JSON`.

Example:

```bash
mkdir -p local/transport/logs
TEXTS_JSONL=local/transport/inputs/preview_texts.jsonl \
  sbatch script/transport/slurm/build_preview.sbatch
```
