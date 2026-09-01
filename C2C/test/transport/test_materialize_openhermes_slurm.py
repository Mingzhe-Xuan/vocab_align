import os
import shlex
import subprocess
from pathlib import Path


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


def _job_path():
    return (
        Path(__file__).resolve().parents[2]
        / "script"
        / "transport"
        / "slurm"
        / "materialize_openhermes_500k.sbatch"
    )


def _fixture(tmp_path, python_body):
    root = tmp_path / "C2C"
    python_bin = root / ".venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    python_bin.write_text(python_body, encoding="utf-8")
    python_bin.chmod(0o755)
    return root, python_bin


def test_openhermes_materialization_job_is_locked_and_partition_agnostic():
    source = _job_path().read_text(encoding="utf-8")
    result = subprocess.run(
        ["bash", "-n", str(_job_path())], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition" not in source
    assert "teknium/OpenHermes-2.5" in source
    assert "05c3557e57b6dd1d0e0cb8369ba53b43e15fd10b" in source
    assert "--sample-count 500000" in source
    assert "--seed 42" in source
    assert "--dev-fraction 0.01" in source


def test_openhermes_materialization_job_forwards_paths_and_failure(tmp_path):
    root, python_bin = _fixture(
        tmp_path,
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-c" ]]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$CAPTURE_PATH"\n'
        'exit "${BUILDER_EXIT:-0}"\n',
    )
    capture = root / "capture.txt"
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_SUBMIT_DIR": _bash_path(root),
            "PYTHON_BIN": _bash_path(python_bin),
            "RECORDS_OUTPUT": _bash_path(root / "records.jsonl"),
            "MANIFEST_OUTPUT": _bash_path(root / "manifest.json"),
            "DATASET_CACHE_DIR": _bash_path(root / "cache"),
            "RESOURCE_TIME_BIN": _bash_path(root / "missing-time"),
            "CAPTURE_PATH": _bash_path(capture),
            "BUILDER_EXIT": "7",
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
    assert process.returncode == 7
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == [
        "-m",
        "script.dataset.materialize_transport_corpus",
        "--dataset",
    ]
    assert (
        arguments[arguments.index("--records-output") + 1]
        == environment["RECORDS_OUTPUT"]
    )
    assert (
        arguments[arguments.index("--manifest-output") + 1]
        == environment["MANIFEST_OUTPUT"]
    )
