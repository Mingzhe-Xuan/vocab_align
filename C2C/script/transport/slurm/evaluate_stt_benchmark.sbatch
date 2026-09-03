#!/usr/bin/env bash
#SBATCH --job-name=stt-benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=local/transport/logs/stt-benchmark-%j.out
#SBATCH --error=local/transport/logs/stt-benchmark-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the C2C repository root}"
config_path="${1:?usage: sbatch evaluate_stt_benchmark.sbatch CONFIG.yaml}"
mkdir -p local/transport/logs

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

.venv-smoke-cu128/bin/python -m script.evaluation.unified_evaluator \
  --config "${config_path}"
