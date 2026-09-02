import os
import shlex
import subprocess
from pathlib import Path

import yaml

from rosetta.transport.config import TransportConfig


def _bash_path(path: Path) -> str:
    process = subprocess.run(
        ["bash", "-lc", f"cygpath -u {shlex.quote(str(path))}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert process.returncode == 0, process.stderr
    return [line for line in process.stdout.splitlines() if line.strip()][-1]


def _job_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "script"
        / "transport"
        / "slurm"
        / "smoke_real_models.sbatch"
    )


def test_real_smoke_job_has_locked_resources_and_no_partition():
    job = _job_path()
    process = subprocess.run(
        ["bash", "-n", str(job)], capture_output=True, text=True, check=False
    )
    assert process.returncode == 0, process.stderr
    source = job.read_text(encoding="utf-8")
    assert "#SBATCH --partition" not in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --mem=192G" in source
    assert "#SBATCH --time=04:00:00" in source
    assert "qwen3_8b_to_mistral_nemo_instruct_2407_smoke.yaml" in source
    assert "--require-cuda" in source
    assert "--require-locked-runtime" in source
    assert "--runtime-profile" in source
    assert "--min-gpu-memory-gib" in source
    assert 'HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"' in source


def test_real_smoke_recipe_is_pinned_and_minimal():
    recipe = (
        Path(__file__).resolve().parents[2]
        / "recipe"
        / "transport_recipe"
        / "qwen3_8b_to_mistral_nemo_instruct_2407_smoke.yaml"
    )
    config = TransportConfig.from_dict(
        yaml.safe_load(recipe.read_text(encoding="utf-8"))
    )
    assert config.generation == {
        "do_sample": False,
        "max_new_tokens": 2,
        "temperature": 1.0,
    }
    assert config.output_schema == "stt-smoke-v2"
    assert config.output_path.endswith("qwen3_8b_to_mistral_nemo_openhermes_500k.npz")
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '"accelerate==1.9.0"' in pyproject


def test_real_smoke_job_forwards_pinned_defaults(tmp_path):
    root = tmp_path / "C2C"
    python_bin = root / ".venv" / "bin" / "python"
    config = root / "smoke.yaml"
    artifact = root / "artifact.npz"
    output = root / "result.json"
    capture = root / "capture.txt"
    python_bin.parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    config.write_text("schema_version: 1\n", encoding="utf-8")
    artifact.write_bytes(b"fixture")
    python_bin.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    python_bin.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_SUBMIT_DIR": _bash_path(root),
            "PYTHON_BIN": _bash_path(python_bin),
            "SMOKE_CONFIG": _bash_path(config),
            "ARTIFACT_PATH": _bash_path(artifact),
            "SMOKE_OUTPUT": _bash_path(output),
            "CAPTURE_PATH": _bash_path(capture),
            "CODE_VERSION": "validation-commit",
            "RUNTIME_PROFILE": "blackwell-cu128",
            "RESOURCE_TIME_BIN": _bash_path(root / "missing-time"),
        }
    )
    process = subprocess.run(
        ["bash", str(_job_path())],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert process.returncode == 0, process.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == ["-m", "script.transport.smoke_stt", "--config"]
    assert arguments[arguments.index("--config") + 1] == _bash_path(config)
    assert arguments[arguments.index("--artifact") + 1] == _bash_path(artifact)
    assert arguments[arguments.index("--output") + 1] == _bash_path(output)
    assert arguments[arguments.index("--code-version") + 1] == "validation-commit"
    assert arguments[arguments.index("--runtime-profile") + 1] == "blackwell-cu128"
    assert arguments[arguments.index("--min-gpu-memory-gib") + 1] == "20"
