#!/usr/bin/env bash
#SBATCH --job-name=stt-mmlu-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=local/transport/logs/stt-mmlu-%j.out
#SBATCH --error=local/transport/logs/stt-mmlu-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?submit this job from the C2C repository root}"
mkdir -p local/transport/logs local/transport/evaluation/mmlu-redux

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

.venv-smoke-cu128/bin/python -m script.evaluation.unified_evaluator \
  --config recipe/eval_recipe/stt_mmlu_redux.yaml
