import pytest

from rosetta.transport.candidate_graph import (
    CandidateGraphError,
    EdgeSource,
    accumulate_byte_span_counts,
    build_candidate_graph,
)


def test_special_and_duplicate_exact_precedence(TinyTokenizer):
    source = TinyTokenizer(
        {"<eos-a>": 0, "x": 1},
        {"x": [(1, 0, 1)]},
        specials=("<eos-a>",),
        eos_token_id=0,
    )
    target = TinyTokenizer(
        {"<eos-b>": 0, "x": 1, "x-duplicate": 2},
        {"x": [(1, 0, 1)]},
        specials=("<eos-b>",),
        eos_token_id=0,
    )
    # Give ID 2 the same bytes without relying on its visible spelling.
    target._by_id[2] = "x"
    graph = build_candidate_graph(source, target, ["x"], required_source_ids=[0, 1])
    by_source = {source_id: [] for source_id in (0, 1)}
    for edge in graph.edges:
        by_source[edge.source_id].append(edge)
    assert [(edge.target_id, edge.source) for edge in by_source[0]] == [
        (0, EdgeSource.SPECIAL)
    ]
    assert [(edge.target_id, edge.source) for edge in by_source[1]] == [
        (1, EdgeSource.EXACT_BYTE),
        (2, EdgeSource.EXACT_BYTE),
    ]


def test_special_literal_fallback_uses_only_ordinary_target_tokens(TinyTokenizer):
    control = "<control>"
    source = TinyTokenizer(
        {control: 0},
        {control: [(0, 0, len(control))]},
        specials=(control,),
    )
    target = TinyTokenizer(
        {"<": 0, "control": 1, ">": 2, "<bos>": 3},
        {control: [(0, 0, 1), (1, 1, 8), (2, 8, 9)]},
        specials=("<bos>",),
        bos_token_id=3,
    )
    graph = build_candidate_graph(
        source,
        target,
        [],
        required_source_ids=[0],
        required_target_ids=[0, 1, 2],
        special_literal_fallback=True,
    )
    assert [(edge.target_id, edge.source) for edge in graph.edges] == [
        (0, EdgeSource.SPECIAL_LITERAL),
        (1, EdgeSource.SPECIAL_LITERAL),
        (2, EdgeSource.SPECIAL_LITERAL),
    ]

    target._pieces[control] = [(3, 0, len(control))]
    with pytest.raises(CandidateGraphError, match="no ordinary literal"):
        build_candidate_graph(
            source,
            target,
            [],
            required_source_ids=[0],
            special_literal_fallback=True,
        )


def test_span_overlap_counts_multibyte_content(TinyTokenizer):
    text = "a中🙂é"
    source = TinyTokenizer(
        {"a中": 0, "🙂é": 1},
        {text: [(0, 0, 2), (1, 2, 5)]},
    )
    target = TinyTokenizer(
        {"a": 0, "中🙂": 1, "é": 2},
        {text: [(0, 0, 1), (1, 1, 3), (2, 3, 5)]},
    )
    counts = accumulate_byte_span_counts(source, target, [text])
    assert counts[0] == {0: 1, 1: 3}
    assert counts[1] == {1: 4, 2: 3}


def test_ann_fallback_and_required_target_support(TinyTokenizer):
    source = TinyTokenizer({"missing": 0}, {"q": [(0, 0, 1)]})
    target = TinyTokenizer({"target": 0, "other": 1}, {"q": [(1, 0, 1)]})
    graph = build_candidate_graph(
        source,
        target,
        [],
        required_source_ids=[0],
        required_target_ids=[0],
        ann_fallback=lambda source_id, raw: [(0, 0.7)],
    )
    assert graph.edges[0].source == EdgeSource.ANN
    with pytest.raises(CandidateGraphError, match="target token"):
        build_candidate_graph(
            source,
            target,
            [],
            required_source_ids=[0],
            required_target_ids=[1],
            ann_fallback=lambda source_id, raw: [(0, 0.7)],
        )


def test_missing_safe_fallback_fails(TinyTokenizer):
    source = TinyTokenizer({"missing": 0}, {"q": [(0, 0, 1)]})
    target = TinyTokenizer({"target": 0}, {"q": [(0, 0, 1)]})
    with pytest.raises(CandidateGraphError, match="no exact/span edge"):
        build_candidate_graph(source, target, [], required_source_ids=[0])


def test_required_target_rescue_adds_observed_span_edges(TinyTokenizer):
    text = "ab"
    source = TinyTokenizer({"ab": 0}, {text: [(0, 0, 2)]})
    target = TinyTokenizer(
        {"ab-unused": 0, "a": 1, "b": 2},
        {text: [(1, 0, 1), (2, 1, 2)]},
    )
    target._by_id[0] = "ab"
    graph = build_candidate_graph(
        source,
        target,
        [text],
        required_source_ids=[0],
        required_target_ids=[1, 2],
    )
    assert [(edge.target_id, edge.source) for edge in graph.edges] == [
        (0, EdgeSource.EXACT_BYTE),
        (1, EdgeSource.BYTE_SPAN),
        (2, EdgeSource.BYTE_SPAN),
    ]


def test_duplicate_ann_edge_is_rejected(TinyTokenizer):
    source = TinyTokenizer({"missing": 0}, {"q": [(0, 0, 1)]})
    target = TinyTokenizer({"target": 0}, {"q": [(0, 0, 1)]})
    with pytest.raises(CandidateGraphError, match="duplicate"):
        build_candidate_graph(
            source,
            target,
            [],
            required_source_ids=[0],
            ann_fallback=lambda source_id, raw: [(0, 0.7), (0, 0.6)],
        )


def test_ann_augments_exact_source_and_preserves_exact_precedence(TinyTokenizer):
    source = TinyTokenizer({"x": 0}, {"x": [(0, 0, 1)]})
    target = TinyTokenizer(
        {"x": 0, "y": 1},
        {"x": [(0, 0, 1)]},
    )
    graph = build_candidate_graph(
        source,
        target,
        ["x"],
        required_source_ids=[0],
        required_target_ids=[0, 1],
        ann_fallback=lambda source_id, raw: [(0, 0.9), (1, 0.5)],
    )
    assert [(edge.target_id, edge.source) for edge in graph.edges] == [
        (0, EdgeSource.EXACT_BYTE),
        (1, EdgeSource.ANN),
    ]

    exact_only = build_candidate_graph(
        source,
        target,
        ["x"],
        required_source_ids=[0],
        ann_fallback=lambda source_id, raw: [],
    )
    assert len(exact_only.edges) == 1
    assert exact_only.edges[0].source == EdgeSource.EXACT_BYTE
