from pathlib import Path

import pytest
import yaml

from rosetta.transport.config import (
    ConfigError,
    DataSpec,
    ModelSpec,
    PENDING_CHECKPOINT,
    SpecialTokenPolicy,
    TransportConfig,
    TransportConstructionSpec,
    TransportInferenceSpec,
)


REVISION_A = "a" * 40
REVISION_B = "b" * 40


def _payload():
    return {
        "schema_version": 1,
        "source": {
            "name": "source/model",
            "revision": REVISION_A,
            "tokenizer_revision": REVISION_A,
            "dtype": "bfloat16",
            "device_map": "auto",
        },
        "target": {
            "name": "target/model",
            "revision": REVISION_B,
            "tokenizer_revision": REVISION_B,
            "dtype": "bfloat16",
            "device_map": {"": 1},
            "checkpoint": PENDING_CHECKPOINT,
        },
        "data": {
            "dataset": "openhermes",
            "revision": REVISION_A,
            "build_splits": ["transport_train"],
            "dev_fraction": 0.01,
        },
        "seed": 42,
        "output_path": "local/transport/artifacts/main.npz",
        "output_schema": "stt-result-v1",
        "construction": {
            "epsilon": 0.5,
            "tolerance": 0.002,
            "max_iter": 10_000,
            "smoothing": 1e-8,
        },
        "transport": {"tau": 0.7, "causal_shift": True, "source_top_m": 128},
        "generation": {"max_new_tokens": 64, "do_sample": False},
    }


def test_config_stable_round_trip():
    config = TransportConfig.from_dict(_payload())
    encoded = config.to_json()
    assert TransportConfig.from_json(encoded) == config
    assert TransportConfig.from_json(encoded).to_json() == encoded


@pytest.mark.parametrize("field", ["seed", "output_path"])
def test_config_rejects_missing_required_root_fields(field):
    payload = _payload()
    del payload[field]
    with pytest.raises(ConfigError, match="missing required"):
        TransportConfig.from_dict(payload)


def test_config_rejects_unpinned_revision_and_test_split():
    payload = _payload()
    payload["source"]["revision"] = "main"
    with pytest.raises(ConfigError, match="pinned"):
        TransportConfig.from_dict(payload)

    payload = _payload()
    payload["data"]["build_splits"] = ["benchmark_test"]
    with pytest.raises(ConfigError, match="test splits"):
        TransportConfig.from_dict(payload)

    payload = _payload()
    payload["data"]["revision"] = "main"
    with pytest.raises(ConfigError, match="dataset revision"):
        TransportConfig.from_dict(payload)


def test_pending_checkpoint_cannot_be_loaded():
    model = ModelSpec.from_dict(_payload()["target"])
    assert model.checkpoint_available is False
    with pytest.raises(ConfigError, match="pending"):
        model.require_checkpoint()


def test_data_and_seed_validation_are_not_coerced():
    with pytest.raises(ConfigError, match="dev_fraction"):
        DataSpec("dataset", REVISION_A, dev_fraction=0).validate()
    payload = _payload()
    payload["seed"] = True
    with pytest.raises(ConfigError, match="seed"):
        TransportConfig.from_dict(payload)


def test_transport_inference_defaults_and_validation():
    payload = _payload()
    del payload["transport"]
    assert TransportConfig.from_dict(payload).transport == TransportInferenceSpec()

    payload = _payload()
    payload["transport"]["causal_shift"] = "yes"
    with pytest.raises(ConfigError, match="causal_shift"):
        TransportConfig.from_dict(payload)

    payload = _payload()
    payload["transport"]["tau"] = float("inf")
    with pytest.raises(ConfigError, match="finite"):
        TransportConfig.from_dict(payload)


def test_transport_construction_defaults_and_validation():
    payload = _payload()
    del payload["construction"]
    assert TransportConfig.from_dict(payload).construction == (
        TransportConstructionSpec()
    )

    for field, value in (
        ("epsilon", 0),
        ("tolerance", float("inf")),
        ("smoothing", True),
        ("max_iter", False),
    ):
        payload = _payload()
        payload["construction"][field] = value
        with pytest.raises(ConfigError, match=field):
            TransportConfig.from_dict(payload)


def test_pinned_recipe_explicitly_enables_causal_shift():
    recipe = (
        Path(__file__).resolve().parents[2]
        / "recipe"
        / "transport_recipe"
        / ("qwen3_8b_to_mistral_nemo_instruct_2407.yaml")
    )
    config = TransportConfig.from_dict(
        yaml.safe_load(recipe.read_text(encoding="utf-8"))
    )
    assert config.transport.causal_shift is True
    assert config.transport.tau == 1.0
    assert config.transport.source_top_m is None
    assert config.construction == TransportConstructionSpec(
        epsilon=0.5,
        tolerance=0.002,
        max_iter=10_000,
        smoothing=1e-8,
    )


def _recipe(name):
    path = Path(__file__).resolve().parents[2] / "recipe" / "transport_recipe" / name
    return TransportConfig.from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_reverse_and_second_model_pair_recipes_pin_independent_directions():
    forward = _recipe("qwen3_8b_to_mistral_nemo_instruct_2407.yaml")
    reverse = _recipe("mistral_nemo_to_qwen3_8b.yaml")
    second = _recipe("qwen3_8b_to_deepseek_r1_distill_llama_8b.yaml")

    assert reverse.source.name == forward.target.name
    assert reverse.target.name == forward.source.name
    assert reverse.source.revision == forward.target.revision
    assert reverse.target.revision == forward.source.revision
    assert reverse.source.tokenizer_fingerprint == forward.target.tokenizer_fingerprint
    assert reverse.target.tokenizer_fingerprint == forward.source.tokenizer_fingerprint
    assert reverse.expected_artifact_shape == (151643, 131072)
    assert forward.expected_artifact_shape == (131069, 151669)
    assert reverse.output_path != forward.output_path
    assert second.target.name == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    assert second.expected_artifact_shape == (128000, 151669)
    assert len({forward.output_path, reverse.output_path, second.output_path}) == 3
    assert forward.special_tokens == reverse.special_tokens == second.special_tokens
    assert forward.special_tokens == SpecialTokenPolicy()


def test_recipe_fingerprints_shapes_and_special_policy_are_strict():
    payload = _payload()
    payload["source"]["tokenizer_fingerprint"] = "not-a-sha"
    with pytest.raises(ConfigError, match="fingerprint"):
        TransportConfig.from_dict(payload)

    payload = _payload()
    payload["expected_artifact_shape"] = [2, 0]
    with pytest.raises(ConfigError, match="artifact_shape"):
        TransportConfig.from_dict(payload)

    payload = _payload()
    payload["special_tokens"] = {"target_support": "all_tokens"}
    with pytest.raises(ConfigError, match="special token policy"):
        TransportConfig.from_dict(payload)
