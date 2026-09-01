import numpy as np
import pytest

from rosetta.transport.marginals import MarginalError, estimate_token_marginal


def test_marginal_uses_canonical_content_and_filters_zero_support(TinyTokenizer):
    tokenizer = TinyTokenizer(
        {"a": 0, "b": 1, "unused": 2, "<pad>": 3},
        {"ab": [(0, 0, 1), (1, 1, 2)]},
        specials=("<pad>",),
        pad_token_id=3,
    )
    marginal = estimate_token_marginal(tokenizer, ["ab", "ab"], smoothing=0.5)
    assert marginal.active_ids == (0, 1)
    assert np.all(marginal.probabilities[list(marginal.active_ids)] > 0)
    assert marginal.probabilities[2] == marginal.probabilities[3] == 0
    assert marginal.probabilities.sum() == pytest.approx(1.0)
    assert all(call["add_special_tokens"] is False for call in tokenizer.call_kwargs)


def test_marginal_special_pseudocount_and_invalid_empty_input(TinyTokenizer):
    tokenizer = TinyTokenizer(
        {"a": 0, "<eos>": 1},
        {"a": [(0, 0, 1)]},
        specials=("<eos>",),
        eos_token_id=1,
    )
    marginal = estimate_token_marginal(
        tokenizer, ["a"], special_pseudocounts={"eos": 1.0}
    )
    assert marginal.active_ids == (0, 1)
    np.testing.assert_allclose(marginal.probabilities, [0.5, 0.5])
    with pytest.raises(MarginalError, match="no active"):
        estimate_token_marginal(tokenizer, [])
