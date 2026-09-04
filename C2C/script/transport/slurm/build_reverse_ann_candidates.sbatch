#!/usr/bin/env bash
#SBATCH --job-name=stt-reverse-ann
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=local/transport/logs/stt-reverse-ann-%j.out
#SBATCH --error=local/transport/logs/stt-reverse-ann-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:?submit from the C2C repository root}"
cd "$submit_dir"
test -f pyproject.toml

python_bin="${PYTHON_BIN:-.venv/bin/python}"
output="${ANN_CANDIDATES_JSON:-local/transport/artifacts/mistral_nemo_to_qwen3_8b_ann.json}"

test -x "$python_bin"
test ! -e "$output"
mkdir -p "$(dirname "$output")" local/transport/logs

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" PYTHONPATH=. \
  "$python_bin" -m script.transport.build_ann_candidates \
    --source mistralai/Mistral-Nemo-Instruct-2407 \
    --source-revision 04d8a90549d23fc6bd7f642064003592df51e9b3 \
    --target Qwen/Qwen3-8B \
    --target-revision b968826d9c46dd6066d109eabc6255188de91218 \
    --output "$output" \
    --dimension "${ANN_DIMENSION:-256}" \
    --signature-bits "${ANN_SIGNATURE_BITS:-16}" \
    --top-k "${ANN_TOP_K:-8}" \
    --pool-size "${ANN_POOL_SIZE:-128}" \
    --seed "${ANN_SEED:-42}" \
    --code-version "${CODE_VERSION:-$(git rev-parse HEAD)}"
