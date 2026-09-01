"""Build a small Qwen-to-Qwen vocabulary transport artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from rosetta.transport.vocab_transport import build_small_transport, save_transport


DEFAULT_TEXTS = [
    "Explain why the sky appears blue in one sentence.",
    "What is 17 × 23? Show the essential calculation.",
    "请用一句话解释为什么天空看起来是蓝色的。",
    "新加坡的官方语言有哪些？请简洁回答。",
    "Python code: for i in range(3): print(i)",
    "Unicode audit: café, naïve, 中文，emoji 🙂, tabs\tand newlines\nend.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--target", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--texts-jsonl", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local/transport/qwen25_05b_to_qwen3_06b.small.json"),
    )
    return parser.parse_args()


def load_texts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_TEXTS
    texts = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        text = value if isinstance(value, str) else value.get("text")
        if not isinstance(text, str):
            raise ValueError(f"Line {line_number} must be a JSON string or contain a text field")
        texts.append(text)
    return texts


def main() -> None:
    args = parse_args()
    source = AutoTokenizer.from_pretrained(args.source, use_fast=True)
    target = AutoTokenizer.from_pretrained(args.target, use_fast=True)
    artifact = build_small_transport(source, target, load_texts(args.texts_jsonl))
    save_transport(artifact, args.output)
    print(json.dumps(artifact.audit, ensure_ascii=False, indent=2))
    print(f"Saved {len(artifact.columns)} sparse columns to {args.output}")


if __name__ == "__main__":
    main()
