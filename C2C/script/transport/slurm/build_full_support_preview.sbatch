#!/usr/bin/env bash
#SBATCH --job-name=stt-full-preview
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=local/transport/logs/stt-full-preview-%j.out
#SBATCH --error=local/transport/logs/stt-full-preview-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:?submit from the C2C repository root}"
cd "$submit_dir"
test -f pyproject.toml

python_bin="${PYTHON_BIN:-.venv/bin/python}"
texts_jsonl="${TEXTS_JSONL:-local/transport/inputs/preview_texts.jsonl}"
ann_candidates="${ANN_CANDIDATES_JSON:-local/transport/artifacts/qwen3_8b_to_mistral_nemo_ann.json}"
artifact="${ARTIFACT_PATH:-local/transport/artifacts/qwen3_8b_to_mistral_nemo_full_support_preview.npz}"
audit_json="${AUDIT_JSON:-local/transport/audits/qwen3_8b_to_mistral_nemo_full_support_preview.json}"
audit_markdown="${AUDIT_MARKDOWN:-local/transport/audits/qwen3_8b_to_mistral_nemo_full_support_preview.md}"
smoothing="${TRANSPORT_SMOOTHING:-1e-8}"

test -x "$python_bin"
test -f "$texts_jsonl"
test -f "$ann_candidates"
mkdir -p "$(dirname "$artifact")" "$(dirname "$audit_json")" \
  "$(dirname "$audit_markdown")" local/transport/logs

"$python_bin" -c \
  'import math, sys; value = float(sys.argv[1]); raise SystemExit(0 if math.isfinite(value) and value > 0 else 2)' \
  "$smoothing"

arguments=(
  -m script.transport.build_vocab_transport
  --source Qwen/Qwen3-8B
  --source-revision b968826d9c46dd6066d109eabc6255188de91218
  --target mistralai/Mistral-Nemo-Instruct-2407
  --target-revision 04d8a90549d23fc6bd7f642064003592df51e9b3
  --texts-jsonl "$texts_jsonl"
  --ann-candidates-json "$ann_candidates"
  --artifact "$artifact"
  --audit-json "$audit_json"
  --audit-markdown "$audit_markdown"
  --epsilon "${TRANSPORT_EPSILON:-0.5}"
  --tolerance "${TRANSPORT_TOLERANCE:-1e-9}"
  --max-iter "${TRANSPORT_MAX_ITER:-10000}"
  --smoothing "$smoothing"
  --seed 42
  --code-version "${CODE_VERSION:-$(git rev-parse HEAD)}"
)

if [[ "${RESUME:-0}" == "1" ]]; then
  arguments+=(--resume)
fi

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" PYTHONPATH=. \
  "$python_bin" "${arguments[@]}"
