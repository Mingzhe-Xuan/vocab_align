import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rosetta.transport.artifact import artifact_from_dense
from rosetta.transport.config import TransportConfig
from rosetta.transport.metrics import TransportMetrics
from rosetta.transport.soft_transport import SoftTransportStats
from rosetta.transport.wrapper import TransportGenerationOutput
from script.transport.smoke_stt import SmokeError, build_smoke_report, save_smoke_report


REVISION_A = "a" * 40
REVISION_B = "b" * 40


def _config():
    return TransportConfig.from_dict(
        {
            "schema_version": 1,
            "source": {
                "name": "source/model",
                "revision": REVISION_A,
                "tokenizer_revision": REVISION_A,
            },
            "target": {
                "name": "target/model",
                "revision": REVISION_B,
                "tokenizer_revision": REVISION_B,
            },
            "data": {
                "dataset": "tiny",
                "revision": REVISION_A,
                "build_splits": ["transport_train"],
                "dev_fraction": 0.1,
            },
            "seed": 42,
            "output_path": "local/transport/artifacts/tiny.npz",
            "output_schema": "stt-result-v1",
            "transport": {
                "tau": 0.7,
                "causal_shift": True,
                "source_top_m": None,
            },
            "generation": {"max_new_tokens": 2, "do_sample": False},
        }
    )


def _artifact():
    return artifact_from_dense(
        np.eye(2),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        {
            "schema_version": 1,
            "source_fingerprint": "source",
            "target_fingerprint": "target",
            "input_fingerprint": "artifact-input",
            "build_config": {"epsilon": 0.5},
            "seed": 42,
            "code_version": "artifact-code",
        },
    )


class SourceTokenizer:
    def __call__(self, text, **kwargs):
        assert text == "hello"
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": True}
        return {
            "input_ids": torch.tensor([[0, 1]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class TargetTokenizer:
    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "decoded:" + ",".join(str(value) for value in token_ids)


class DummyWrapper:
    def __init__(self):
        self.source_model = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros((2, 3)))
        )
        self.artifact = _artifact()
        self.calls = []

    def generate(self, input_ids, **kwargs):
        self.calls.append((input_ids.clone(), kwargs))
        metrics = TransportMetrics(
            source_seconds=0.1,
            transport_seconds=0.2,
            receiver_prefill_seconds=0.3,
            decode_seconds=0.4,
            total_seconds=1.0,
            source_input_tokens=2,
            virtual_tokens=2,
            output_tokens=2,
            peak_memory_bytes=None,
        )
        stats = SoftTransportStats(
            retained_mass=torch.ones((1, 2)),
            dropped_top_m_mass=torch.zeros((1, 2)),
            active_support_mass=torch.ones((1, 2)),
            top_m=None,
        )
        return TransportGenerationOutput(
            sequences=torch.tensor([[3, 4]]),
            virtual_prompt_shape=(1, 2, 3),
            stats=stats,
            metrics=metrics,
        )


def test_build_and_atomically_save_smoke_report(tmp_path):
    wrapper = DummyWrapper()
    report = build_smoke_report(
        wrapper,
        SourceTokenizer(),
        TargetTokenizer(),
        prompt="hello",
        generation=_config().generation,
        config=_config(),
        code_version="test-code",
    )
    assert report["schema_version"] == 1
    assert len(report["input_fingerprint"]) == 64
    assert report["artifact"]["shape"] == [2, 2]
    assert report["shapes"] == {
        "source_input_ids": [1, 2],
        "virtual_prompt": [1, 2, 3],
        "receiver_output_ids": [1, 2],
    }
    assert report["transport_quality"]["retained_mass"]["minimum"] == 1.0
    assert report["metrics"]["memory_status"] == "unavailable"
    assert report["outputs"] == [{"receiver_token_ids": [3, 4], "text": "decoded:3,4"}]
    assert wrapper.calls[0][1]["return_transport_output"] is True

    output = tmp_path / "nested" / "smoke.json"
    save_smoke_report(report, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not output.with_name(output.name + ".partial").exists()


def test_smoke_rejects_unknown_generation_fields():
    with pytest.raises(SmokeError, match="unsupported"):
        build_smoke_report(
            DummyWrapper(),
            SourceTokenizer(),
            TargetTokenizer(),
            prompt="hello",
            generation={"max_new_tokens": 1, "communication_temperature": 2.0},
            config=_config(),
            code_version="test",
        )


def test_smoke_cli_help_does_not_load_remote_models():
    process = subprocess.run(
        [sys.executable, "-m", "script.transport.smoke_stt", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert "--artifact" in process.stdout
