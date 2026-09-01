import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from rosetta.transport.baseline import BaselineError, freeze_baseline
from rosetta.transport.config import TransportConfig


def _config():
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load(
        (root / "recipe/transport_recipe/qwen3_8b_to_mistral_nemo_instruct_2407.yaml").read_text(
            encoding="utf-8"
        )
    )
    return TransportConfig.from_dict(payload)


def test_baseline_keeps_canonical_and_rendered_inputs_separate():
    messages = [{"role": "user", "content": "你好"}]
    prompts = {"source": "<src>你好", "target": "<tgt>你好"}
    runtime = {"gpu": {"status": "unavailable"}, "dependencies": {}}
    first = freeze_baseline(
        _config(), messages, prompts, code_version="commit", runtime=runtime
    )
    second = freeze_baseline(
        _config(), messages, prompts, code_version="commit", runtime=runtime
    )

    assert first.to_json() == second.to_json()
    assert first.canonical_messages == messages
    assert first.rendered_prompts == prompts
    assert first.checkpoint["status"] == "pending"
    assert first.models["source"]["tokenizer_revision"] == "b968826d9c46dd6066d109eabc6255188de91218"


def test_baseline_rejects_invalid_messages_and_prompts():
    with pytest.raises(BaselineError, match="role and content"):
        freeze_baseline(
            _config(),
            [{"role": "user", "content": "x", "extra": "bad"}],
            {"source": "s", "target": "t"},
            code_version="commit",
        )
    with pytest.raises(BaselineError, match="source and target"):
        freeze_baseline(
            _config(),
            [{"role": "user", "content": "x"}],
            {"source": "s"},
            code_version="commit",
        )


def test_freeze_baseline_cli(tmp_path):
    root = Path(__file__).resolve().parents[2]
    messages = tmp_path / "messages.json"
    prompts = tmp_path / "prompts.json"
    output = tmp_path / "baseline.json"
    messages.write_text('[{"role":"user","content":"hello"}]', encoding="utf-8")
    prompts.write_text('{"source":"src hello","target":"tgt hello"}', encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "script.transport.freeze_baseline",
            "--config",
            "recipe/transport_recipe/qwen3_8b_to_mistral_nemo_instruct_2407.yaml",
            "--messages",
            str(messages),
            "--rendered-prompts",
            str(prompts),
            "--output",
            str(output),
            "--code-version",
            "test-commit",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["code_version"] == "test-commit"
    assert snapshot["runtime"]["gpu"]["status"] == "unavailable"
