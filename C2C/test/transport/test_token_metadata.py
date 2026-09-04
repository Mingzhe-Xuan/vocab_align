from rosetta.transport.token_metadata import (
    encode_with_byte_spans,
    exact_byte_matches,
    ordinary_bytes_index,
    special_id_to_kind,
    token_raw_bytes,
)


class TinyTokenizer:
    is_fast = True

    def __init__(self, vocab, pieces=None, specials=(), **special_ids):
        self.name_or_path = "tiny"
        self._vocab = vocab
        self._by_id = {value: key for key, value in vocab.items()}
        self._pieces = pieces or {}
        self.all_special_tokens = list(specials)
        for name, value in special_ids.items():
            setattr(self, name, value)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        return self._by_id[token_id]

    def __call__(self, text, **kwargs):
        entries = self._pieces[text]
        return {
            "input_ids": [item[0] for item in entries],
            "offset_mapping": [(item[1], item[2]) for item in entries],
        }


def test_raw_bytes_support_utf8_and_byte_level_bpe():
    tokenizer = TinyTokenizer({"中": 0, "Ā": 1, "ðŁĻĤ": 2})
    assert token_raw_bytes(tokenizer, 0) == "中".encode("utf-8")
    assert token_raw_bytes(tokenizer, 1) == b"\x00"
    assert token_raw_bytes(tokenizer, 2) == "🙂".encode("utf-8")


def test_character_offsets_are_converted_to_byte_offsets():
    text = "中🙂é"
    tokenizer = TinyTokenizer(
        {"中": 0, "🙂": 1, "é": 2},
        {text: [(0, 0, 1), (1, 1, 2), (2, 2, 4)]},
    )
    assert encode_with_byte_spans(tokenizer, text) == [
        (0, 0, 3),
        (1, 3, 7),
        (2, 7, 10),
    ]


def test_special_tokens_are_classified_and_excluded_from_exact_index():
    tokenizer = TinyTokenizer(
        {"ordinary": 0, "<eos>": 1},
        specials=("<eos>",),
        eos_token_id=1,
    )
    assert special_id_to_kind(tokenizer) == {1: "eos"}
    assert ordinary_bytes_index(tokenizer) == {b"ordinary": [0]}


def test_backend_special_added_tokens_are_classified_when_not_in_public_list():
    class AddedToken:
        special = True

        def __str__(self):
            return "<control>"

    class Backend:
        @staticmethod
        def get_added_tokens_decoder():
            return {1: AddedToken()}

    tokenizer = TinyTokenizer({"ordinary": 0, "<control>": 1})
    tokenizer.backend_tokenizer = Backend()

    assert tokenizer.all_special_tokens == []
    assert special_id_to_kind(tokenizer) == {1: "special"}
    assert ordinary_bytes_index(tokenizer) == {b"ordinary": [0]}


def test_same_token_id_with_different_bytes_is_not_an_exact_match():
    source = TinyTokenizer({"a": 0})
    target = TinyTokenizer({"b": 0})
    assert exact_byte_matches(source, target) == {}
