import subprocess
from pathlib import Path


def test_reverse_transport_job_is_cpu_only_and_uses_directional_paths():
    job = (
        Path(__file__).resolve().parents[2]
        / "script"
        / "transport"
        / "slurm"
        / "reverse_formal_transport.sbatch"
    )
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
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --partition" not in source
    assert "#SBATCH --mem=16G" in source
    assert "#SBATCH --time=01:00:00" in source
    assert "qwen3_8b_to_mistral_nemo_openhermes_500k.npz" in source
    assert "mistral_nemo_to_qwen3_8b_openhermes_500k.npz" in source
    assert "script.transport.reverse_vocab_transport" in source
