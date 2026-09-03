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
    assert recipe["model"]["generation_config"]["max_new_tokens"] == 64
    assert recipe["model"]["sender_generation_config"] == {
        "do_sample": False,
        "max_new_tokens": 64,
        "temperature": 1.0,
    }
    assert recipe["eval"]["use_cot"] is True
    assert "planner-thinker" in recipe["output"]["output_dir"]
    assert recipe["eval"]["limit"] == 5


def test_cross_benchmark_smoke_recipes_are_exact_and_isolated():
    expected = {
        "stt_gsm8k.yaml": ("gsm8k", 3, 64),
        "stt_math500.yaml": ("math-500", 3, 128),
        "stt_longbench_qasper.yaml": ("longbench", 1, 32),
    }
    output_dirs = set()
    for filename, (dataset, limit, max_new_tokens) in expected.items():
        recipe = yaml.safe_load(
            Path("recipe/eval_recipe", filename).read_text(encoding="utf-8")
        )
        assert recipe["model"]["source_device_map"] == "cpu"
        assert recipe["model"]["target_device_map"] == "auto"
        assert "approximation" not in recipe["model"]
        assert recipe["model"]["generation_config"]["do_sample"] is False
        assert recipe["model"]["generation_config"]["max_new_tokens"] == max_new_tokens
        assert recipe["eval"]["dataset"] == dataset
        assert recipe["eval"]["limit"] == limit
        assert len(recipe["eval"]["dataset_revision"]) == 40
        output_dirs.add(recipe["output"]["output_dir"])
    assert len(output_dirs) == len(expected)
    for filename in ("stt_math500.yaml", "stt_longbench_qasper.yaml"):
        recipe = yaml.safe_load(
            Path("recipe/eval_recipe", filename).read_text(encoding="utf-8")
        )
        assert recipe["eval"]["data_format"] in {"json", "parquet"}
        assert len(recipe["eval"]["data_file_sha256"]) == 64

    script = Path("script/transport/slurm/evaluate_stt_benchmark.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:1" in script
    assert ".venv-smoke-cu128/bin/python" in script
    assert 'config_path="${1:?' in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "uv " not in script

    prompt = yaml.safe_load(
        Path("recipe/eval_recipe/longbench_qasper_prompt.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(prompt) == {"qasper"}
    assert "{context}" in prompt["qasper"]
    assert "{input}" in prompt["qasper"]
