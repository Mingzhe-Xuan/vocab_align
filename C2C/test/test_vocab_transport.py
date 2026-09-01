from rosetta.transport.vocab_transport import build_small_transport


class TinyTokenizer:
    is_fast = True

    def __init__(self, vocab, pieces, special_tokens=()):
        self.name_or_path = "tiny"
        self._vocab = vocab
        self._pieces = pieces
        self.all_special_tokens = list(special_tokens)

    def __len__(self):
        return len(self._vocab)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        return next(token for token, value in self._vocab.items() if value == token_id)

    def __call__(self, text, **kwargs):
        entries = self._pieces[text]
        return {
            "input_ids": [item[0] for item in entries],
            "offset_mapping": [(item[1], item[2]) for item in entries],
        }


def test_exact_and_span_columns_are_normalized():
    source = TinyTokenizer(
        {"a": 0, "bc": 1, "<eos>": 2},
        {"abc": [(0, 0, 1), (1, 1, 3)]},
        ["<eos>"],
    )
    target = TinyTokenizer(
        {"a": 0, "b": 1, "c": 2, "<eos>": 3},
        {"abc": [(0, 0, 1), (1, 1, 2), (2, 2, 3)]},
        ["<eos>"],
    )

    artifact = build_small_transport(source, target, ["abc"])
    columns = {column.source_id: column for column in artifact.columns}
    assert columns[0].rule == "exact_byte"
    assert columns[0].target_ids == [0]
    assert columns[1].rule == "byte_span"
    assert columns[1].target_ids == [1, 2]
    assert columns[1].weights == [0.5, 0.5]
    assert artifact.audit["max_column_sum_error"] == 0.0
    assert artifact.audit["nonnegative"] is True
