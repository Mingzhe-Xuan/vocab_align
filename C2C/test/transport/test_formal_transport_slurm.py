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


def _job_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "script"
        / "transport"
        / "slurm"
        / "build_formal_transport.sbatch"
    )


def _fixture(tmp_path: Path, python_body: str):
    root = tmp_path / "C2C"
    python_bin = root / ".venv" / "bin" / "python"
    records = root / "records.jsonl"
    manifest = root / "manifest.json"
    ann = root / "ann.json"
    python_bin.parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    records.write_text('{"conversations":[]}\n', encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    ann.write_text("{}\n", encoding="utf-8")
    python_bin.write_text(python_body, encoding="utf-8")
    python_bin.chmod(0o755)
    return root, python_bin, records, manifest, ann


def _environment(root, python_bin, records, manifest, ann, capture):
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_SUBMIT_DIR": _bash_path(root),
            "PYTHON_BIN": _bash_path(python_bin),
            "RECORDS_JSONL": _bash_path(records),
            "MANIFEST_JSON": _bash_path(manifest),
            "ANN_CANDIDATES_JSON": _bash_path(ann),
            "CAPTURE_PATH": _bash_path(capture),
            "CODE_VERSION": "accepted-commit",
            "RESOURCE_TIME_BIN": _bash_path(root / "missing-time"),
        }
    )
    return environment


def test_formal_transport_job_is_locked_and_partition_agnostic():
    job = _job_path()
    process = subprocess.run(
        ["bash", "-n", str(job)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert process.returncode == 0, process.stderr
    source = job.read_text(encoding="utf-8")
    assert "#SBATCH --partition" not in source
    assert "#SBATCH --mem=64G" in source
    assert "#SBATCH --time=24:00:00" in source
    assert "openhermes-500k.jsonl" in source
    assert "openhermes-500k.json" in source
    assert "BUILD_SPLIT:-transport_train" in source
    assert "TRANSPORT_TOLERANCE:-2e-3" in source
    assert "TRANSPORT_SMOOTHING:-1e-8" in source
    assert "--texts-jsonl" not in source


def test_formal_transport_job_forwards_manifest_inputs_and_defaults(tmp_path):
    root, python_bin, records, manifest, ann = _fixture(
        tmp_path,
        '#!/usr/bin/env bash\nif [[ "$1" == "-c" ]]; then exit 0; fi\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
    )
    capture = root / "capture.txt"
    environment = _environment(root, python_bin, records, manifest, ann, capture)
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
    assert arguments[:3] == ["-m", "script.transport.build_vocab_transport", "--source"]
    assert "b968826d9c46dd6066d109eabc6255188de91218" in arguments
    assert "04d8a90549d23fc6bd7f642064003592df51e9b3" in arguments
    assert arguments[arguments.index("--records-jsonl") + 1] == _bash_path(records)
    assert arguments[arguments.index("--manifest-json") + 1] == _bash_path(manifest)
    assert arguments[arguments.index("--build-split") + 1] == "transport_train"
    assert arguments[arguments.index("--ann-candidates-json") + 1] == _bash_path(ann)
    assert arguments[arguments.index("--epsilon") + 1] == "0.5"
    assert arguments[arguments.index("--tolerance") + 1] == "2e-3"
    assert arguments[arguments.index("--max-iter") + 1] == "10000"
    assert arguments[arguments.index("--smoothing") + 1] == "1e-8"
    assert arguments[arguments.index("--code-version") + 1] == "accepted-commit"


def test_formal_transport_job_forwards_overrides_and_resume(tmp_path):
    root, python_bin, records, manifest, ann = _fixture(
        tmp_path,
        '#!/usr/bin/env bash\nif [[ "$1" == "-c" ]]; then exit 0; fi\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
    )
    capture = root / "capture.txt"
    environment = _environment(root, python_bin, records, manifest, ann, capture)
    environment.update(
        {
            "BUILD_SPLIT": "transport_dev",
            "TRANSPORT_EPSILON": "0.25",
            "TRANSPORT_TOLERANCE": "1e-3",
            "TRANSPORT_MAX_ITER": "4321",
            "TRANSPORT_SMOOTHING": "2e-8",
            "RESUME": "1",
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
    assert arguments[arguments.index("--build-split") + 1] == "transport_dev"
    assert arguments[arguments.index("--epsilon") + 1] == "0.25"
    assert arguments[arguments.index("--tolerance") + 1] == "1e-3"
    assert arguments[arguments.index("--max-iter") + 1] == "4321"
    assert arguments[arguments.index("--smoothing") + 1] == "2e-8"
    assert arguments[-1] == "--resume"


def test_formal_transport_job_propagates_builder_failure(tmp_path):
    root, python_bin, records, manifest, ann = _fixture(
        tmp_path,
        '#!/usr/bin/env bash\nif [[ "$1" == "-c" ]]; then exit 0; fi\nexit 7\n',
    )
    environment = _environment(
        root, python_bin, records, manifest, ann, root / "unused.txt"
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
