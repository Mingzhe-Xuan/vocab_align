"""Compare two fast byte-level tokenizers without downloading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from rosetta.transport.token_metadata import (
    encode_with_byte_spans,
    iter_token_metadata,
    special_id_to_token,
    tokenizer_fingerprint,
)


DEFAULT_TEXTS = [
    "Explain why the sky appears blue in one sentence.",
    "What is 17 × 23? Show the essential calculation.",
    "请用一句话解释为什么天空看起来是蓝色的。",
    "新加坡的官方语言有哪些？请简洁回答。",
    "Python code: for i in range(3): print(i)",
    "Unicode audit: café, naïve, 中文，emoji 🙂, tabs\tand newlines\nend.",
]


def added_tokens(tokenizer: Any) -> dict[int, dict[str, Any]]:
    decoder = tokenizer.backend_tokenizer.get_added_tokens_decoder()
    return {
        int(token_id): {"token": str(token), "special": bool(token.special)}
        for token_id, token in decoder.items()
    }


def ordinary_vocab(tokenizer: Any) -> dict[str, int]:
    return {
        item.token: item.token_id
        for item in iter_token_metadata(tokenizer)
        if not item.is_special
    }


def fingerprint(vocab: dict[str, int]) -> str:
    payload = json.dumps(sorted(vocab.items(), key=lambda item: item[1]), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_spans(tokenizer: Any, text: str) -> list[tuple[int, int, int]]:
    return encode_with_byte_spans(tokenizer, text)


def reported_revision(tokenizer: Any, requested_revision: str | None) -> str | None:
    """Report the resolved commit, retaining an explicitly requested pin.

    Some local-cache and mirror code paths omit Transformers' private
    ``_commit_hash`` field even though ``revision`` was passed to the loader.
    In that case the explicit immutable SHA remains the authoritative pin.
    """
    return tokenizer.init_kwargs.get("_commit_hash") or requested_revision


def code_version() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def audit_input_fingerprint(
    source_fingerprint: str, target_fingerprint: str, texts: list[str]
) -> str:
    payload = json.dumps(
        {
            "source_fingerprint": source_fingerprint,
            "target_fingerprint": target_fingerprint,
            "texts": texts,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-revision')
    parser.add_argument('--target-revision')
    parser.add_argument("--source", default="Qwen/Qwen3-8B")
    parser.add_argument("--target", default="mistralai/Mistral-Nemo-Instruct-2407")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-version")
    args = parser.parse_args()

    source_kwargs = {'use_fast': True}
    target_kwargs = {'use_fast': True}
    if args.source_revision:
        source_kwargs['revision'] = args.source_revision
    if args.target_revision:
        target_kwargs['revision'] = args.target_revision
    source = AutoTokenizer.from_pretrained(args.source, **source_kwargs)
    target = AutoTokenizer.from_pretrained(args.target, **target_kwargs)
    source_vocab = ordinary_vocab(source)
    target_vocab = ordinary_vocab(target)
    source_bytes: dict[bytes, list[int]] = defaultdict(list)
    target_bytes: dict[bytes, list[int]] = defaultdict(list)
    for item in iter_token_metadata(source):
        if not item.is_special:
            source_bytes[item.raw_bytes].append(item.token_id)
    for item in iter_token_metadata(target):
        if not item.is_special:
            target_bytes[item.raw_bytes].append(item.token_id)

    shared_bytes = set(source_bytes) & set(target_bytes)
    source_exact_ids = {token_id for value in shared_bytes for token_id in source_bytes[value]}
    target_exact_ids = {token_id for value in shared_bytes for token_id in target_bytes[value]}
    source_added = added_tokens(source)
    target_added = added_tokens(target)

    sample_rows = []
    source_occurrences: Counter[int] = Counter()
    exact_occurrences = 0
    total_occurrences = 0
    for text in DEFAULT_TEXTS:
        source_spans = token_spans(source, text)
        target_spans = token_spans(target, text)
        source_ids = [token_id for token_id, _, _ in source_spans]
        target_ids = [token_id for token_id, _, _ in target_spans]
        source_occurrences.update(source_ids)
        total_occurrences += len(source_ids)
        exact_occurrences += sum(token_id in source_exact_ids for token_id in source_ids)
        sample_rows.append(
            {
                "text": text,
                "source_length": len(source_ids),
                "target_length": len(target_ids),
                "length_ratio_target_over_source": len(target_ids) / max(len(source_ids), 1),
                "same_id_sequence": source_ids == target_ids,
                "source_tokens": source.convert_ids_to_tokens(source_ids),
                "target_tokens": target.convert_ids_to_tokens(target_ids),
            }
        )

    source_special = {token: token_id for token_id, token in special_id_to_token(source).items()}
    target_special = {token: token_id for token_id, token in special_id_to_token(target).items()}
    common_control = sorted(set(source_special) & set(target_special))
    source_fingerprint = tokenizer_fingerprint(source)
    target_fingerprint = tokenizer_fingerprint(target)
    report = {
        "schema_version": 1,
        "input_fingerprint": audit_input_fingerprint(
            source_fingerprint, target_fingerprint, DEFAULT_TEXTS
        ),
        "build_config": {
            "source": args.source,
            "source_revision": args.source_revision,
            "target": args.target,
            "target_revision": args.target_revision,
            "text_set": "built-in-multilingual-audit-v1",
        },
        "seed": args.seed,
        "code_version": args.code_version or code_version(),
        'requested_revisions': {
            'source': args.source_revision,
            'target': args.target_revision,
        },
        "source": {
            "name": args.source,
            "revision": reported_revision(source, args.source_revision),
            "total_vocab_size": len(source),
            "ordinary_vocab_size": len(source_vocab),
            "added_token_count": len(source_added),
            "ordinary_vocab_fingerprint": fingerprint(source_vocab),
            "tokenizer_fingerprint": source_fingerprint,
        },
        "target": {
            "name": args.target,
            "revision": reported_revision(target, args.target_revision),
            "total_vocab_size": len(target),
            "ordinary_vocab_size": len(target_vocab),
            "added_token_count": len(target_added),
            "ordinary_vocab_fingerprint": fingerprint(target_vocab),
            "tokenizer_fingerprint": target_fingerprint,
        },
        "ordinary_byte_overlap": {
            "shared_unique_byte_strings": len(shared_bytes),
            "source_token_coverage": len(source_exact_ids) / len(source_vocab),
            "target_token_coverage": len(target_exact_ids) / len(target_vocab),
            "source_tokens_with_exact_target": len(source_exact_ids),
            "target_tokens_with_exact_source": len(target_exact_ids),
            "duplicate_byte_strings_source": sum(len(ids) > 1 for ids in source_bytes.values()),
            "duplicate_byte_strings_target": sum(len(ids) > 1 for ids in target_bytes.values()),
        },
        "control_tokens": {
            "common_count": len(common_control),
            "common": [
                {
                    "token": token,
                    "source_id": source_special[token],
                    "target_id": target_special[token],
                }
                for token in common_control
            ],
            "source_only_count": len(set(source_special) - set(target_special)),
            "target_only_count": len(set(target_special) - set(source_special)),
            "source_only": sorted(set(source_special) - set(target_special)),
            "target_only_sample": sorted(set(target_special) - set(source_special))[:40],
        },
        "sample_audit": {
            "text_count": len(DEFAULT_TEXTS),
            "unique_source_tokens": len(source_occurrences),
            "source_occurrence_exact_byte_coverage": exact_occurrences / total_occurrences,
            "mean_length_ratio_target_over_source": sum(
                row["length_ratio_target_over_source"] for row in sample_rows
            ) / len(sample_rows),
            "rows": sample_rows,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report["ordinary_byte_overlap"], **report["sample_audit"]}, ensure_ascii=False, indent=2))
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
