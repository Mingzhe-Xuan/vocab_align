from pathlib import Path

import yaml


def test_transport_evaluation_slurm_uses_locked_gpu_runtime():
    script = Path("script/transport/slurm/evaluate_stt_mmlu_redux.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:1" in script
    assert ".venv-smoke-cu128/bin/python" in script
    assert "script.evaluation.unified_evaluator" in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "TRANSFORMERS_OFFLINE=1" in script
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in script
    assert "uv " not in script

    recipe = yaml.safe_load(
        Path("recipe/eval_recipe/stt_mmlu_redux.yaml").read_text(encoding="utf-8")
    )
    assert recipe["model"]["require_cuda"] is True
    assert recipe["model"]["require_locked_runtime"] is True
    assert recipe["model"]["min_gpu_memory_gib"] == 30.0
    assert recipe["model"]["source_device_map"] == "cpu"
    assert recipe["model"]["target_device_map"] == "auto"
    assert recipe["model"]["generation_config"]["max_new_tokens"] == 16
    assert recipe["eval"]["limit"] == 5
