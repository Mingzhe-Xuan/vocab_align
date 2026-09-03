from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import rosetta.transport.wrapper as wrapper_module
from rosetta.transport.artifact import artifact_from_dense
from rosetta.transport.approximations import precompute_source_values
from rosetta.transport.orf import apply_orf_transport, build_orf_transport_state
from rosetta.transport.wrapper import (
    TrainingFreeTransportModel,
    TransportGenerationOutput,
    TransportModelError,
)


class TinySource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(3))
        self.calls = []

    def forward(self, input_ids, attention_mask, position_ids, **kwargs):
        self.calls.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "attention_mask": attention_mask.clone(),
                "position_ids": position_ids.clone(),
            }
        )
        predicted = (input_ids + 1) % 3
        logits = 8 * torch.nn.functional.one_hot(predicted, 3).float() + self.bias
        return SimpleNamespace(logits=logits)


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, attention_mask, position_ids, **kwargs):
        self.calls.append(input_ids.clone())
        hidden = torch.nn.functional.one_hot((input_ids + 1) % 3, 3).float()
        return SimpleNamespace(last_hidden_state=hidden)


class TinyOrfSource(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head_calls = 0

    def forward(self, *args, **kwargs):
        self.lm_head_calls += 1
        raise AssertionError("ORF must bypass the causal LM head")


class TinyReceiver(torch.nn.Module):
    def __init__(self, *, return_cache=True):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor(
                [[10.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]
            ),
            freeze=False,
        )
        self.config = SimpleNamespace(bos_token_id=0, eos_token_id=4, pad_token_id=0)
        self.calls = []
        self.decode_calls = 0
        self.return_cache = return_cache
        self.generate_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "input_ids": None if input_ids is None else input_ids.clone(),
                "inputs_embeds": (
                    None if inputs_embeds is None else inputs_embeds.clone()
                ),
                "attention_mask": attention_mask.clone(),
                "position_ids": position_ids.clone(),
                "past_key_values": past_key_values,
            }
        )
        batch = attention_mask.shape[0]
        if inputs_embeds is not None:
            length = inputs_embeds.shape[1]
            logits = torch.full((batch, length, 5), -10.0)
            for position in range(length):
                logits[:, position, position + 1] = 10.0
            cache = length
        else:
            schedule = ([4, 1], [4, 4])
            tokens = schedule[min(self.decode_calls, len(schedule) - 1)]
            logits = torch.full((batch, 1, 5), -10.0)
            logits[torch.arange(batch), 0, torch.tensor(tokens[:batch])] = 10.0
            self.decode_calls += 1
            cache = int(past_key_values) + 1
        return SimpleNamespace(
            logits=logits,
            past_key_values=cache if self.return_cache else None,
        )

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        marker = torch.full((kwargs["input_ids"].shape[0], 1), 4, dtype=torch.long)
        return torch.cat((kwargs["input_ids"], marker), dim=1)


def _artifact():
    artifact = artifact_from_dense(
        np.eye(3),
        np.full(3, 1 / 3),
        np.full(3, 1 / 3),
        {
            "schema_version": 1,
            "source_fingerprint": "source",
            "target_fingerprint": "target",
            "build_config": {"epsilon": 0.5},
            "seed": 42,
            "code_version": "test",
        },
    )
    return replace(artifact, target_token_ids=np.array([1, 2, 3]))


def _model(*, causal_shift=True, receiver=None):
    source = TinySource()
    receiver = TinyReceiver() if receiver is None else receiver
    return (
        TrainingFreeTransportModel(
            source,
            receiver,
            _artifact(),
            tau=1.0,
            causal_shift=causal_shift,
        ),
        source,
        receiver,
    )


def test_shifted_virtual_prompt_handles_left_and_right_padding_without_gradients():
    model, source, receiver = _model()
    input_ids = torch.tensor([[0, 1, 0], [0, 1, 2]])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
    virtual = model.build_virtual_prompt(input_ids, mask)

    torch.testing.assert_close(virtual.embeddings[0, 0], receiver.embedding.weight[0])
    torch.testing.assert_close(virtual.embeddings[1, 1], receiver.embedding.weight[0])
    assert torch.count_nonzero(virtual.embeddings[0, 2]) == 0
    assert torch.count_nonzero(virtual.embeddings[1, 0]) == 0
    expected = torch.softmax(torch.tensor([0.0, 8.0, 0.0]), dim=-1)
    expected_embedding = expected @ receiver.embedding.weight[1:4]
    torch.testing.assert_close(virtual.embeddings[0, 1], expected_embedding)
    left_padded_expected = torch.softmax(torch.tensor([0.0, 0.0, 8.0]), dim=-1)
    torch.testing.assert_close(
        virtual.embeddings[1, 2],
        left_padded_expected @ receiver.embedding.weight[1:4],
    )
    torch.testing.assert_close(
        virtual.position_ids, torch.tensor([[0, 1, 1], [0, 0, 1]])
    )
    assert source.calls[0]["grad_enabled"] is False
    assert source.bias.grad is None
    assert not hasattr(model, "optimizer")


def test_no_shift_uses_same_position_logits_and_single_token_is_supported():
    model, _, receiver = _model(causal_shift=False)
    virtual = model.build_virtual_prompt(torch.tensor([[2]]))
    expected = torch.softmax(torch.tensor([8.0, 0.0, 0.0]), dim=-1)
    torch.testing.assert_close(
        virtual.embeddings[0, 0], expected @ receiver.embedding.weight[1:4]
    )


def test_hard_top_m_and_precomputed_modes_follow_their_oracles():
    input_ids = torch.tensor([[0, 1]])
    receiver = TinyReceiver()
    exact = TrainingFreeTransportModel(
        TinySource(), receiver, _artifact(), tau=1.0, causal_shift=False
    ).build_virtual_prompt(input_ids)

    top_m = TrainingFreeTransportModel(
        TinySource(),
        receiver,
        _artifact(),
        tau=1.0,
        causal_shift=False,
        approximation_mode="top_m",
        source_top_m=3,
    ).build_virtual_prompt(input_ids)
    torch.testing.assert_close(top_m.embeddings, exact.embeddings)
    torch.testing.assert_close(top_m.stats.dropped_top_m_mass, torch.zeros((1, 2)))

    hard = TrainingFreeTransportModel(
        TinySource(),
        receiver,
        _artifact(),
        tau=1.0,
        causal_shift=False,
        approximation_mode="hard",
    ).build_virtual_prompt(input_ids)
    torch.testing.assert_close(
        hard.embeddings,
        torch.stack((receiver.embedding.weight[2], receiver.embedding.weight[3]))[None],
    )

    source_values = precompute_source_values(_artifact(), receiver.embedding.weight)
    precomputed = TrainingFreeTransportModel(
        TinySource(),
        receiver,
        _artifact(),
        tau=1.0,
        causal_shift=False,
        approximation_mode="precomputed",
        precomputed_source_values=source_values,
    ).build_virtual_prompt(input_ids)
    torch.testing.assert_close(precomputed.embeddings, exact.embeddings)


def test_exact_transport_chunks_long_source_queries_without_changing_results(
    monkeypatch,
):
    seen_lengths = []
    original = wrapper_module.transport_embeddings

    def tracked_transport(logits, *args, **kwargs):
        seen_lengths.append(logits.shape[1])
        return original(logits, *args, **kwargs)

    monkeypatch.setattr(wrapper_module, "transport_embeddings", tracked_transport)
    input_ids = torch.tensor([[0, 1, 2] * 21 + [0, 1]])
    model = TrainingFreeTransportModel(
        TinySource(), TinyReceiver(), _artifact(), tau=1.0, causal_shift=False
    )
    virtual = model.build_virtual_prompt(input_ids)

    assert seen_lengths == [32, 32, 1]
    assert virtual.embeddings.shape == (1, 65, 2)
    assert virtual.stats.retained_mass.shape == (1, 65)
    torch.testing.assert_close(virtual.stats.retained_mass, torch.ones((1, 65)))


def test_orf_mode_uses_backbone_only_and_reports_stats_unavailable():
    source = TinyOrfSource()
    receiver = TinyReceiver()
    artifact = _artifact()
    output_weight = torch.eye(3)
    state = build_orf_transport_state(
        output_weight,
        None,
        artifact,
        receiver.embedding.weight,
        feature_count=12,
        tau=1.0,
        seed=42,
    )
    input_ids = torch.tensor([[0, 2]])
    model = TrainingFreeTransportModel(
        source,
        receiver,
        artifact,
        tau=1.0,
        causal_shift=False,
        approximation_mode="orf",
        orf_state=state,
    )
    virtual = model.build_virtual_prompt(input_ids)
    hidden = torch.nn.functional.one_hot((input_ids + 1) % 3, 3).float()
    expected = apply_orf_transport(
        hidden,
        state,
        source_fingerprint="source",
        target_fingerprint="target",
    )
    torch.testing.assert_close(virtual.embeddings, expected)
    assert virtual.stats is None
    assert source.lm_head_calls == 0
    assert len(source.model.calls) == 1


def test_generate_uses_last_active_prefill_logits_then_receiver_cache_only():
    model, source, receiver = _model()
    sequences = model.generate(
        torch.tensor([[0, 1, 0], [0, 1, 2]]),
        source_attention_mask=torch.tensor([[1, 1, 0], [0, 1, 1]]),
        max_new_tokens=3,
        eos_token_id=4,
        pad_token_id=0,
    )
    torch.testing.assert_close(sequences, torch.tensor([[2, 4, 0], [3, 1, 4]]))
    assert len(source.calls) == 1
    assert receiver.calls[0]["inputs_embeds"] is not None
    assert all(call["input_ids"] is not None for call in receiver.calls[1:])
    torch.testing.assert_close(
        receiver.calls[1]["attention_mask"],
        torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]]),
    )
    torch.testing.assert_close(
        receiver.calls[1]["position_ids"], torch.tensor([[2], [2]])
    )
    assert receiver.calls[1]["past_key_values"] == 3
    assert receiver.calls[2]["past_key_values"] == 4


def test_planner_context_is_aligned_before_receiver_native_prompt():
    model, _, receiver = _model(causal_shift=False)
    source_ids = torch.tensor([[0, 1]])
    receiver_ids = torch.tensor([[3, 4]])
    output = model.generate(
        source_ids,
        receiver_input_ids=receiver_ids,
        receiver_attention_mask=torch.ones_like(receiver_ids),
        max_new_tokens=1,
        return_transport_output=True,
    )
    call = receiver.calls[0]
    assert call["input_ids"] is None
    assert call["inputs_embeds"].shape == (1, 4, 2)
    expected_first = torch.softmax(torch.tensor([0.0, 8.0, 0.0]), dim=-1)
    expected_second = torch.softmax(torch.tensor([0.0, 0.0, 8.0]), dim=-1)
    torch.testing.assert_close(
        call["inputs_embeds"][0, 0],
        expected_first @ receiver.embedding.weight[1:4],
    )
    torch.testing.assert_close(
        call["inputs_embeds"][0, 1],
        expected_second @ receiver.embedding.weight[1:4],
    )
    torch.testing.assert_close(
        call["inputs_embeds"][0, 2:], receiver.embedding(receiver_ids)[0]
    )
    torch.testing.assert_close(
        call["attention_mask"], torch.ones((1, 4), dtype=torch.long)
    )
    torch.testing.assert_close(call["position_ids"], torch.tensor([[0, 1, 2, 3]]))
    assert output.aligned_sender_shape == (1, 2, 2)
    assert output.receiver_prompt_shape == (1, 2, 2)
    assert output.virtual_prompt_shape == (1, 4, 2)
    assert output.metrics.virtual_tokens == 2
    assert output.metrics.receiver_prompt_tokens == 2
    assert output.metrics.to_dict()["prefill_tokens"] == 4


def test_structured_generation_reports_shapes_quality_and_segmented_metrics():
    model, _, _ = _model()
    output = model.generate(
        torch.tensor([[0, 1]]),
        max_new_tokens=2,
        eos_token_id=None,
        return_transport_output=True,
    )
    assert isinstance(output, TransportGenerationOutput)
    assert output.virtual_prompt_shape == (1, 2, 2)
    assert output.sequences.shape == (1, 2)
    assert output.metrics.source_input_tokens == 2
    assert output.metrics.virtual_tokens == 2
    assert output.metrics.output_tokens == 2
    assert output.metrics.peak_memory_bytes is None
    assert output.metrics.total_seconds == pytest.approx(
        output.metrics.source_seconds
        + output.metrics.transport_seconds
        + output.metrics.receiver_prefill_seconds
        + output.metrics.decode_seconds
    )
    torch.testing.assert_close(output.stats.retained_mass, torch.ones((1, 2)))


def test_receiver_only_path_is_an_exact_generate_pass_through():
    model, source, receiver = _model()
    receiver_ids = torch.tensor([[1, 2]])
    mask = torch.ones_like(receiver_ids)
    actual = model.generate(
        receiver_input_ids=receiver_ids,
        receiver_attention_mask=mask,
        transport=False,
        max_new_tokens=2,
        eos_token_id=4,
        pad_token_id=0,
    )
    torch.testing.assert_close(actual, torch.tensor([[1, 2, 4]]))
    assert not source.calls
    assert receiver.generate_calls[0]["input_ids"] is receiver_ids
    assert receiver.generate_calls[0]["attention_mask"] is mask


@pytest.mark.parametrize(
    "input_ids,mask,message",
    [
        (torch.tensor([1, 2]), None, "shape"),
        (torch.tensor([[1, 2]]), torch.tensor([[1, 0, 1]]), "match"),
        (torch.tensor([[1, 2, 0]]), torch.tensor([[1, 0, 1]]), "contiguous"),
        (torch.tensor([[1, 2]]), torch.tensor([[0, 0]]), "active token"),
    ],
)
def test_invalid_prompt_shapes_and_masks_fail(input_ids, mask, message):
    model, _, _ = _model()
    with pytest.raises(TransportModelError, match=message):
        model.build_virtual_prompt(input_ids, mask)


def test_missing_start_token_and_cache_fail_explicitly():
    receiver = TinyReceiver(return_cache=False)
    receiver.config.bos_token_id = None
    with pytest.raises(TransportModelError, match="start token"):
        TrainingFreeTransportModel(TinySource(), receiver, _artifact(), tau=1.0)
    model = TrainingFreeTransportModel(
        TinySource(), receiver, _artifact(), tau=1.0, receiver_start_token_id=0
    )
    with pytest.raises(TransportModelError, match="KV cache"):
        model.prefill(torch.tensor([[0, 1]]))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"tau": 0}, "tau"),
        ({"tau": 1.0, "source_top_m": 0}, "source_top_m"),
        ({"tau": 1.0, "receiver_start_token_id": 99}, "vocabulary"),
        ({"tau": 1.0, "approximation_mode": "unknown"}, "mode"),
        ({"tau": 1.0, "approximation_mode": "top_m"}, "source_top_m"),
        (
            {"tau": 1.0, "approximation_mode": "precomputed"},
            "precomputed_source_values",
        ),
        ({"tau": 1.0, "approximation_mode": "orf"}, "orf_state"),
    ],
)
def test_invalid_transport_parameters_fail_at_construction(kwargs, message):
    with pytest.raises(TransportModelError, match=message):
        TrainingFreeTransportModel(TinySource(), TinyReceiver(), _artifact(), **kwargs)
