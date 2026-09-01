from script.transport.compare_tokenizers import audit_input_fingerprint, reported_revision


class FakeTokenizer:
    def __init__(self, commit_hash=None):
        self.init_kwargs = {"_commit_hash": commit_hash}


def test_reported_revision_prefers_resolved_commit():
    assert reported_revision(FakeTokenizer("resolved"), "requested") == "resolved"


def test_reported_revision_retains_explicit_pin_when_cache_omits_private_hash():
    pinned = "a" * 40
    assert reported_revision(FakeTokenizer(), pinned) == pinned
    assert reported_revision(FakeTokenizer(), None) is None


def test_audit_input_fingerprint_is_stable_and_input_sensitive():
    first = audit_input_fingerprint("source", "target", ["中文", "🙂"])
    second = audit_input_fingerprint("source", "target", ["中文", "🙂"])
    assert first == second
    assert first != audit_input_fingerprint("source", "other", ["中文", "🙂"])
