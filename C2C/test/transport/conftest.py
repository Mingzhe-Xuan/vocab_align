import pytest


class _TinyTokenizer:
    is_fast = True

    def __init__(self, vocab, pieces, specials=(), **special_ids):
        self.name_or_path = "tiny"
        self._vocab = vocab
        self._by_id = {value: key for key, value in vocab.items()}
        self._pieces = pieces
        self.all_special_tokens = list(specials)
        self.call_kwargs = []
        for name, value in special_ids.items():
            setattr(self, name, value)

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        return self._by_id[token_id]

    def __call__(self, text, **kwargs):
        self.call_kwargs.append(kwargs)
        entries = self._pieces[text]
        return {
            "input_ids": [item[0] for item in entries],
            "offset_mapping": [(item[1], item[2]) for item in entries],
        }


@pytest.fixture
def TinyTokenizer():
    return _TinyTokenizer
