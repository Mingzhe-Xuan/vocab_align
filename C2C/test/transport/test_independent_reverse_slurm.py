import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANN_JOB = ROOT / "script/transport/slurm/build_reverse_ann_candidates.sbatch"
FORMAL_JOB = ROOT / "script/transport/slurm/build_reverse_formal_transport.sbatch"
MISTRAL_REVISION = "04d8a90549d23fc6bd7f642064003592df51e9b3"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


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


def _fake_python(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "-c" ]]; then exit 0; fi\n'
        'printf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def _arguments(capture: Path) -> list[str]:
    return capture.read_text(encoding="utf-8").splitlines()


def test_independent_reverse_jobs_are_cpu_only_and_direction_locked():
    for job in (ANN_JOB, FORMAL_JOB):
        process = subprocess.run(
            ["bash", "-n", str(job)], capture_output=True, text=True, check=False
        )
        assert process.returncode == 0, process.stderr
        source = job.read_text(encoding="utf-8")
        assert "#SBATCH --partition" not in source
        assert "#SBATCH --gres" not in source
        assert "mistralai/Mistral-Nemo-Instruct-2407" in source
        assert "Qwen/Qwen3-8B" in source
        assert MISTRAL_REVISION in source
        assert QWEN_REVISION in source
        assert "qwen3_8b_to_mistral_nemo_ann.json" not in source
        assert "reverse_vocab_transport" not in source

    formal = FORMAL_JOB.read_text(encoding="utf-8")
    assert "openhermes-500k.jsonl" in formal
    assert "openhermes-500k.json" in formal
    assert "BUILD_SPLIT:-transport_train" in formal
    assert "TRANSPORT_TOLERANCE:-2e-3" in formal
    assert "TRANSPORT_SMOOTHING:-1e-8" in formal
    assert "artifact.shape == (151643, 131072)" in formal
    assert '"derivation" not in artifact.metadata' in formal


def test_reverse_ann_job_forwards_independent_direction(tmp_path):
    root = tmp_path / "C2C"
    python_bin = root / ".venv/bin/python"
    capture = root / "ann-arguments.txt"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    _fake_python(python_bin)
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_SUBMIT_DIR": _bash_path(root),
            "PYTHON_BIN": _bash_path(python_bin),
            "CAPTURE_PATH": _bash_path(capture),
            "CODE_VERSION": "reverse-test",
        }
    )

    process = subprocess.run(
        ["bash", str(ANN_JOB)],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert process.returncode == 0, process.stderr
    arguments = _arguments(capture)
    assert arguments[:4] == [
        "-m",
        "script.transport.build_ann_candidates",
        "--source",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ]
    assert arguments[arguments.index("--target") + 1] == "Qwen/Qwen3-8B"
    assert arguments[arguments.index("--source-revision") + 1] == MISTRAL_REVISION
    assert arguments[arguments.index("--target-revision") + 1] == QWEN_REVISION


def test_reverse_formal_job_forwards_manifest_ann_and_solver(tmp_path):
    root = tmp_path / "C2C"
    python_bin = root / ".venv/bin/python"
    capture = root / "formal-arguments.txt"
    records = root / "records.jsonl"
    manifest = root / "manifest.json"
    ann = root / "reverse-ann.json"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    records.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    ann.write_text("{}\n", encoding="utf-8")
    _fake_python(python_bin)
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_SUBMIT_DIR": _bash_path(root),
            "PYTHON_BIN": _bash_path(python_bin),
            "CAPTURE_PATH": _bash_path(capture),
            "RECORDS_JSONL": _bash_path(records),
            "MANIFEST_JSON": _bash_path(manifest),
            "ANN_CANDIDATES_JSON": _bash_path(ann),
            "RESOURCE_TIME_BIN": _bash_path(root / "missing-time"),
            "CODE_VERSION": "reverse-test",
        }
    )

    process = subprocess.run(
        ["bash", str(FORMAL_JOB)],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert process.returncode == 0, process.stderr
    arguments = _arguments(capture)
    assert arguments[:4] == [
        "-m",
        "script.transport.build_vocab_transport",
        "--source",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ]
    assert arguments[arguments.index("--target") + 1] == "Qwen/Qwen3-8B"
    assert arguments[arguments.index("--records-jsonl") + 1] == _bash_path(records)
    assert arguments[arguments.index("--manifest-json") + 1] == _bash_path(manifest)
    assert arguments[arguments.index("--ann-candidates-json") + 1] == _bash_path(ann)
    assert arguments[arguments.index("--epsilon") + 1] == "0.5"
    assert arguments[arguments.index("--tolerance") + 1] == "2e-3"
    assert arguments[arguments.index("--max-iter") + 1] == "10000"
    assert arguments[arguments.index("--smoothing") + 1] == "1e-8"
