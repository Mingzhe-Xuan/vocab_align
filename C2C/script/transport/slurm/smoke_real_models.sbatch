#!/usr/bin/env bash
#SBATCH --job-name=stt-real-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=local/transport/logs/stt-real-smoke-%j.out
#SBATCH --error=local/transport/logs/stt-real-smoke-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:?submit from the C2C repository root}"
cd "$submit_dir"
test -f pyproject.toml

python_bin="${PYTHON_BIN:-.venv/bin/python}"
config="${SMOKE_CONFIG:-recipe/transport_recipe/qwen3_8b_to_mistral_nemo_instruct_2407_smoke.yaml}"
artifact="${ARTIFACT_PATH:-local/transport/artifacts/qwen3_8b_to_mistral_nemo_openhermes_500k.npz}"
output="${SMOKE_OUTPUT:-local/transport/smoke/qwen3_8b_to_mistral_nemo_openhermes_500k.json}"
prompt="${SMOKE_PROMPT:-Explain gravity in one sentence.}"

test -x "$python_bin"
test -f "$config"
test -f "$artifact"
test ! -e "$output"
test ! -e "$output.partial"
mkdir -p "$(dirname "$output")" local/transport/logs

resource_time="${RESOURCE_TIME_BIN:-/usr/bin/time}"
runner=()
if [[ -x "$resource_time" ]]; then
  runner=("$resource_time" -v)
fi

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
PYTHONPATH=. "${runner[@]}" "$python_bin" \
  -m script.transport.smoke_stt \
  --config "$config" \
  --artifact "$artifact" \
  --prompt "$prompt" \
  --output "$output" \
  --code-version "${CODE_VERSION:-$(git rev-parse HEAD)}" \
  --require-cuda \
  --require-locked-runtime \
  --runtime-profile "${RUNTIME_PROFILE:-project-cu124}" \
  --min-gpu-memory-gib "${MIN_GPU_MEMORY_GIB:-20}"
