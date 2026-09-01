import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from rosetta.transport.ann_candidates import (
    AnnCandidateError,
    ByteLshConfig,
    _candidate_pool,
    build_bidirectional_lsh_candidates,
)
from script.transport.build_ann_candidates import save_candidates


class TinyTokenizer:
    is_fast = True

    def __init__(self, name, vocab, specials=(), **special_ids):
        self.name_or_path = name
        self._vocab = vocab
        self._by_id = {value: key for key, value in vocab.items()}
        self.all_special_tokens = list(specials)
        for name, value in special_ids.items():
            setattr(self, name, value)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        return self._by_id[token_id]


def _tokenizers():
    source = TinyTokenizer(
        "source",
        {"a": 0, "bc": 1, "中": 2, "<eos>": 3},
        specials=("<eos>",),
        eos_token_id=3,
    )
    target = TinyTokenizer(
        "target",
        {"a": 0, "b": 1, "xy": 2, "🙂": 3, "<bos>": 4},
        specials=("<bos>",),
        bos_token_id=4,
    )
    return source, target


def _payload(seed=7, config=None):
    source, target = _tokenizers()
    return build_bidirectional_lsh_candidates(
        source,
        target,
        config=config
        or ByteLshConfig(dimension=32, signature_bits=8, top_k=1, pool_size=3),
        seed=seed,
        code_version="test-version",
    )


def test_bidirectional_candidates_cover_and_connect_all_ordinary_tokens():
    payload = _payload()
    candidates = payload["candidates"]
    assert set(candidates) == {"0", "1", "2"}
    assert all(
        target_id != 4 for edges in candidates.values() for target_id, _ in edges
    )
    assert {target_id for edges in candidates.values() for target_id, _ in edges} == {
        0,
        1,
        2,
        3,
    }

    adjacency = {}
    for source_id, edges in candidates.items():
        source_node = ("source", int(source_id))
        adjacency.setdefault(source_node, set())
        for target_id, evidence in edges:
            assert evidence > 0
            target_node = ("target", target_id)
            adjacency[source_node].add(target_node)
            adjacency.setdefault(target_node, set()).add(source_node)
    reached = set()
    pending = [("source", payload["coverage"]["source_anchor"])]
    while pending:
        node = pending.pop()
        if node in reached:
            continue
        reached.add(node)
        pending.extend(adjacency[node] - reached)
    assert reached == set(adjacency)
    assert payload["coverage"]["source_tokens_with_candidates"] == 3
    assert payload["coverage"]["target_tokens_with_candidates"] == 4
    assert payload["build_config"]["bridge_evidence"] == pytest.approx(1e-6)


def test_candidates_and_fingerprint_are_deterministic_and_seeded():
    first = _payload(seed=11)
    assert first == _payload(seed=11)
    second = _payload(seed=12)
    assert first["input_fingerprint"] != second["input_fingerprint"]


@pytest.mark.parametrize(
    "config",
    [
        ByteLshConfig(dimension=0),
        ByteLshConfig(min_ngram=3, max_ngram=2),
        ByteLshConfig(max_ngram=256),
        ByteLshConfig(signature_bits=33),
        ByteLshConfig(top_k=3, pool_size=2),
        ByteLshConfig(bridge_evidence=0.0),
        ByteLshConfig(bridge_evidence=1e-5),
    ],
)
def test_invalid_ann_config_is_rejected(config):
    with pytest.raises(AnnCandidateError):
        _payload(config=config)


def test_ann_pool_size_caps_even_a_large_exact_signature_bucket():
    assert _candidate_pool(
        7,
        3,
        {7: [0, 1, 2, 3]},
        {3: [0, 1, 2, 3]},
        pool_size=2,
        maximum_length=3,
    ) == [0, 1]


def test_seed_must_fit_unsigned_64_bits():
    with pytest.raises(AnnCandidateError, match="unsigned 64-bit"):
        _payload(seed=2**64)


def test_candidate_save_is_atomic_and_help_does_not_import_transformers(tmp_path):
    output = tmp_path / "nested" / "candidates.json"
    save_candidates({"value": "中"}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": "中"}
    assert not output.with_name(output.name + ".partial").exists()

    root = Path(__file__).resolve().parents[2]
    process = subprocess.run(
        [sys.executable, "-m", "script.transport.build_ann_candidates", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "--source-revision" in process.stdout


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


def test_ann_slurm_job_is_valid_and_forwards_pinned_revisions(tmp_path):
    root = Path(__file__).resolve().parents[2]
    job = root / "script" / "transport" / "slurm" / "build_ann_candidates.sbatch"
    syntax = subprocess.run(
        ["bash", "-n", str(job)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    assert "#SBATCH --partition" not in job.read_text(encoding="utf-8")

    fake_root = tmp_path / "C2C"
    python_bin = fake_root / ".venv" / "bin" / "python"
    capture = fake_root / "captured.txt"
    python_bin.parent.mkdir(parents=True)
    (fake_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    python_bin.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_PATH"\n',
        encoding="utf-8",
    )
    python_bin.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "SLURM_SUBMIT_DIR": _bash_path(fake_root),
            "PYTHON_BIN": _bash_path(python_bin),
            "CAPTURE_PATH": _bash_path(capture),
            "CODE_VERSION": "test-commit",
        }
    )
    process = subprocess.run(
        ["bash", str(job)],
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert process.returncode == 0, process.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == ["-m", "script.transport.build_ann_candidates", "--source"]
    assert "b968826d9c46dd6066d109eabc6255188de91218" in arguments
    assert "04d8a90549d23fc6bd7f642064003592df51e9b3" in arguments
    assert arguments[arguments.index("--code-version") + 1] == "test-commit"
