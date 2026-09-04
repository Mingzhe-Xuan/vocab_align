#!/usr/bin/env bash
#SBATCH --job-name=stt-reverse-formal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=local/transport/logs/stt-reverse-formal-%j.out
#SBATCH --error=local/transport/logs/stt-reverse-formal-%j.err

set -euo pipefail

submit_dir="${SLURM_SUBMIT_DIR:?submit from the C2C repository root}"
cd "$submit_dir"
test -f pyproject.toml

python_bin="${PYTHON_BIN:-.venv/bin/python}"
records_jsonl="${RECORDS_JSONL:-local/transport/corpora/openhermes-500k.jsonl}"
manifest_json="${MANIFEST_JSON:-local/transport/manifests/openhermes-500k.json}"
build_split="${BUILD_SPLIT:-transport_train}"
ann_candidates="${ANN_CANDIDATES_JSON:-local/transport/artifacts/mistral_nemo_to_qwen3_8b_ann.json}"
artifact="${ARTIFACT_PATH:-local/transport/artifacts/mistral_nemo_to_qwen3_8b_openhermes_500k.npz}"
audit_json="${AUDIT_JSON:-local/transport/audits/mistral_nemo_to_qwen3_8b_openhermes_500k.json}"
audit_markdown="${AUDIT_MARKDOWN:-local/transport/audits/mistral_nemo_to_qwen3_8b_openhermes_500k.md}"
smoothing="${TRANSPORT_SMOOTHING:-1e-8}"

test -x "$python_bin"
test -f "$records_jsonl"
test -f "$manifest_json"
test -f "$ann_candidates"
test ! -e "$artifact"
mkdir -p "$(dirname "$artifact")" "$(dirname "$audit_json")" \
  "$(dirname "$audit_markdown")" local/transport/logs

"$python_bin" -c \
  'import math, sys; value = float(sys.argv[1]); raise SystemExit(0 if math.isfinite(value) and value > 0 else 2)' \
  "$smoothing"

arguments=(
  -m script.transport.build_vocab_transport
  --source mistralai/Mistral-Nemo-Instruct-2407
  --source-revision 04d8a90549d23fc6bd7f642064003592df51e9b3
  --target Qwen/Qwen3-8B
  --target-revision b968826d9c46dd6066d109eabc6255188de91218
  --records-jsonl "$records_jsonl"
  --manifest-json "$manifest_json"
  --build-split "$build_split"
  --ann-candidates-json "$ann_candidates"
  --artifact "$artifact"
  --audit-json "$audit_json"
  --audit-markdown "$audit_markdown"
  --epsilon "${TRANSPORT_EPSILON:-0.5}"
  --tolerance "${TRANSPORT_TOLERANCE:-2e-3}"
  --max-iter "${TRANSPORT_MAX_ITER:-10000}"
  --smoothing "$smoothing"
  --seed 42
  --code-version "${CODE_VERSION:-$(git rev-parse HEAD)}"
)

resource_time="${RESOURCE_TIME_BIN:-/usr/bin/time}"
runner=()
if [[ -x "$resource_time" ]]; then
  runner=("$resource_time" -v)
fi

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" PYTHONPATH=. \
  "${runner[@]}" "$python_bin" "${arguments[@]}"

"$python_bin" -c \
  'import sys; from rosetta.transport.artifact import load_transport_artifact; artifact = load_transport_artifact(sys.argv[1]); assert artifact.shape == (151643, 131072); assert "derivation" not in artifact.metadata' \
  "$artifact"
