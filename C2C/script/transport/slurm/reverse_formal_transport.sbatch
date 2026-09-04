#!/usr/bin/env bash
#SBATCH --job-name=reverse-vocab-transport
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=local/transport/logs/reverse-formal-%j.out
#SBATCH --error=local/transport/logs/reverse-formal-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the C2C directory}"

python_bin="${PYTHON_BIN:-.venv/bin/python}"
input="${FORWARD_ARTIFACT_PATH:-local/transport/artifacts/qwen3_8b_to_mistral_nemo_openhermes_500k.npz}"
artifact="${REVERSE_ARTIFACT_PATH:-local/transport/artifacts/mistral_nemo_to_qwen3_8b_openhermes_500k.npz}"
audit_json="${REVERSE_AUDIT_JSON:-local/transport/audits/mistral_nemo_to_qwen3_8b_openhermes_500k.json}"
audit_markdown="${REVERSE_AUDIT_MARKDOWN:-local/transport/audits/mistral_nemo_to_qwen3_8b_openhermes_500k.md}"
code_version="${CODE_VERSION:-$(git rev-parse HEAD)}"
resource_time="${RESOURCE_TIME_BIN:-/usr/bin/time}"

mkdir -p "$(dirname "${artifact}")" "$(dirname "${audit_json}")" local/transport/logs

"${python_bin}" -c "import numpy; import rosetta.transport"

"${resource_time}" -v "${python_bin}" -m script.transport.reverse_vocab_transport \
  --input "${input}" \
  --artifact "${artifact}" \
  --audit-json "${audit_json}" \
  --audit-markdown "${audit_markdown}" \
  --code-version "${code_version}"
