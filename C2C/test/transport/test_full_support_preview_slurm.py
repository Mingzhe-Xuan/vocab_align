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
        / "build_full_support_preview.sbatch"
    )


def _fixture_root(tmp_path: Path, python_body: str):
    root = tmp_path / "C2C"
    python_bin = root / ".venv" / "bin" / "python"
    texts = root / "local" / "transport" / "inputs" / "preview_texts.jsonl"
    ann = root / "local" / "transport" / "artifacts" / "ann.json"
    python_bin.parent.mkdir(parents=True)
    texts.parent.mkdir(parents=True)
    ann.parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    texts.write_text('{"text":"hello"}\n', encoding="utf-8")
    ann.write_text('{"schema_version":1,"candidates":{}}\n', encoding="utf-8")
    python_bin.write_text(python_body, encoding="utf-8")
    python_bin.chmod(0o755)
    return root, python_bin, texts, ann


def test_full_support_preview_job_has_valid_bash_and_no_partition():
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
    assert "TRANSPORT_SMOOTHING:-1e-8" in source
    assert "--ann-candidates-json" in source


def test_full_support_preview_forwards_locked_inputs_to_stub_python(tmp_path):
    root, python_bin, texts, ann = _fixture_root(
        tmp_path,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
    )
    capture = root / "captured.txt"
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_SUBMIT_DIR": _bash_path(root),
            "PYTHON_BIN": _bash_path(python_bin),
            "TEXTS_JSONL": _bash_path(texts),
            "ANN_CANDIDATES_JSON": _bash_path(ann),
            "CAPTURE_PATH": _bash_path(capture),
            "CODE_VERSION": "test-commit",
            "TRANSPORT_SMOOTHING": "2e-8",
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
    assert arguments[:3] == ["-m", "script.transport.build_vocab_transport", "--source"]
    assert "b968826d9c46dd6066d109eabc6255188de91218" in arguments
    assert "04d8a90549d23fc6bd7f642064003592df51e9b3" in arguments
    assert arguments[arguments.index("--ann-candidates-json") + 1] == _bash_path(ann)
    assert arguments[arguments.index("--smoothing") + 1] == "2e-8"
    assert arguments[arguments.index("--code-version") + 1] == "test-commit"


def test_full_support_preview_propagates_builder_failure(tmp_path):
    root, python_bin, texts, ann = _fixture_root(
        tmp_path,
        "#!/usr/bin/env bash\n" 'if [[ "$1" == "-c" ]]; then exit 0; fi\n' "exit 7\n",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_SUBMIT_DIR": _bash_path(root),
            "PYTHON_BIN": _bash_path(python_bin),
            "TEXTS_JSONL": _bash_path(texts),
            "ANN_CANDIDATES_JSON": _bash_path(ann),
            "CODE_VERSION": "test",
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
