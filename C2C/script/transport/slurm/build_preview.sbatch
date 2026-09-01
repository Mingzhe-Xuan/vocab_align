#!/usr/bin/env bash
#SBATCH --job-name=stt-preview
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=local/transport/logs/stt-preview-%j.out
#SBATCH --error=local/transport/logs/stt-preview-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:?submit from the C2C repository root}"
cd "$submit_dir"
test -f pyproject.toml

python_bin="${PYTHON_BIN:-.venv/bin/python}"
texts_jsonl="${TEXTS_JSONL:-local/transport/inputs/preview_texts.jsonl}"
artifact="${ARTIFACT_PATH:-local/transport/artifacts/qwen3_8b_to_mistral_nemo_preview.npz}"
audit_json="${AUDIT_JSON:-local/transport/audits/qwen3_8b_to_mistral_nemo_preview.json}"
audit_markdown="${AUDIT_MARKDOWN:-local/transport/audits/qwen3_8b_to_mistral_nemo_preview.md}"

test -x "$python_bin"
test -f "$texts_jsonl"
mkdir -p "$(dirname "$artifact")" "$(dirname "$audit_json")" "$(dirname "$audit_markdown")"

arguments=(
  -m script.transport.build_vocab_transport
  --source Qwen/Qwen3-8B
  --source-revision b968826d9c46dd6066d109eabc6255188de91218
  --target mistralai/Mistral-Nemo-Instruct-2407
  --target-revision 04d8a90549d23fc6bd7f642064003592df51e9b3
  --texts-jsonl "$texts_jsonl"
  --artifact "$artifact"
  --audit-json "$audit_json"
  --audit-markdown "$audit_markdown"
  --epsilon "${TRANSPORT_EPSILON:-0.5}"
  --tolerance "${TRANSPORT_TOLERANCE:-1e-9}"
  --max-iter "${TRANSPORT_MAX_ITER:-10000}"
  --smoothing 0
  --seed 42
  --code-version "${CODE_VERSION:-$(git rev-parse HEAD)}"
)

if [[ -n "${ANN_CANDIDATES_JSON:-}" ]]; then
  test -f "$ANN_CANDIDATES_JSON"
  arguments+=(--ann-candidates-json "$ANN_CANDIDATES_JSON")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  arguments+=(--resume)
fi

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" PYTHONPATH=. \
  "$python_bin" "${arguments[@]}"
