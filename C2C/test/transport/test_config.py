import pytest

from rosetta.transport.config import (
    ConfigError,
    DataSpec,
    ModelSpec,
    PENDING_CHECKPOINT,
    TransportConfig,
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
            "build_splits": ["transport_train"],
            "dev_fraction": 0.01,
        },
        "seed": 42,
        "output_path": "local/transport/artifacts/main.npz",
        "output_schema": "stt-result-v1",
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


def test_pending_checkpoint_cannot_be_loaded():
    model = ModelSpec.from_dict(_payload()["target"])
    assert model.checkpoint_available is False
    with pytest.raises(ConfigError, match="pending"):
        model.require_checkpoint()


def test_data_and_seed_validation_are_not_coerced():
    with pytest.raises(ConfigError, match="dev_fraction"):
        DataSpec("dataset", dev_fraction=0).validate()
    payload = _payload()
    payload["seed"] = True
    with pytest.raises(ConfigError, match="seed"):
        TransportConfig.from_dict(payload)
