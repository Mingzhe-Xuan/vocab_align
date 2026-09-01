#!/usr/bin/env bash
#SBATCH --job-name=stt-openhermes-500k
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=local/transport/logs/stt-openhermes-500k-%j.out
#SBATCH --error=local/transport/logs/stt-openhermes-500k-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:?submit from the C2C repository root}"
cd "$submit_dir"
test -f pyproject.toml

python_bin="${PYTHON_BIN:-.venv/bin/python}"
records_output="${RECORDS_OUTPUT:-local/transport/corpora/openhermes-500k.jsonl}"
manifest_output="${MANIFEST_OUTPUT:-local/transport/manifests/openhermes-500k.json}"
cache_dir="${DATASET_CACHE_DIR:-local/transport/datasets/huggingface}"

test -x "$python_bin"
mkdir -p "$(dirname "$records_output")" "$(dirname "$manifest_output")" \
  "$cache_dir" local/transport/logs
"$python_bin" -c 'import datasets; assert datasets.__version__ == "4.0.0"'

arguments=(
  -m script.dataset.materialize_transport_corpus
  --dataset teknium/OpenHermes-2.5
  --dataset-revision 05c3557e57b6dd1d0e0cb8369ba53b43e15fd10b
  --raw-split train
  --cache-dir "$cache_dir"
  --records-output "$records_output"
  --manifest-output "$manifest_output"
  --sample-count 500000
  --seed 42
  --dev-fraction 0.01
)

resource_time="${RESOURCE_TIME_BIN:-/usr/bin/time}"
runner=()
if [[ -x "$resource_time" ]]; then
  runner=("$resource_time" -v)
fi

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" PYTHONPATH=. \
  "${runner[@]}" "$python_bin" "${arguments[@]}"
