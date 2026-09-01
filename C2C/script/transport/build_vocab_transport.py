"""Build, atomically validate, and audit a sparse vocabulary transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from rosetta.transport.artifact import load_transport_artifact, save_transport_artifact
from rosetta.transport.audit import audit_transport_artifact, save_audit
from rosetta.transport.corpus import file_sha256, load_manifest_texts
from rosetta.transport.vocab_transport import build_vocab_transport


class _ToyTokenizer:
    is_fast = True

    def __init__(
        self,
        name: str,
        vocab: dict[str, int],
        pieces: dict[str, list[tuple[int, int, int]]],
    ):
        self.name_or_path = name
        self._vocab = vocab
        self._by_id = {value: key for key, value in vocab.items()}
        self._pieces = pieces
        self.all_special_tokens: list[str] = []

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._by_id[token_id]

    def __call__(self, text: str, **_: Any) -> dict[str, list[Any]]:
        pieces = self._pieces[text]
        return {
            "input_ids": [item[0] for item in pieces],
            "offset_mapping": [(item[1], item[2]) for item in pieces],
        }


def _git_version() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _toy_inputs():
    text = "abcd"
    source = _ToyTokenizer(
        "toy-source", {"ab": 0, "cd": 1}, {text: [(0, 0, 2), (1, 2, 4)]}
    )
    target = _ToyTokenizer(
        "toy-target",
        {"a": 0, "b": 1, "c": 2, "d": 3},
        {text: [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 4)]},
    )
    return (
        source,
        target,
        [text],
        None,
        {"enabled": False, "kind": "toy"},
        {"mode": "toy"},
    )


def _load_ann_candidates(path: Path):
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("ANN candidate JSON root must be an object")
    if "candidates" in payload:
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported ANN candidate schema_version")
        candidates = payload["candidates"]
        required = {
            "input_fingerprint",
            "source_fingerprint",
            "target_fingerprint",
            "build_config",
            "seed",
            "code_version",
            "coverage",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"ANN candidate provenance missing: {sorted(missing)}")
        provenance = {
            key: value for key, value in payload.items() if key != "candidates"
        }
    else:
        candidates = payload
        provenance = {"schema_version": 0, "kind": "legacy-mapping"}
    if not isinstance(candidates, dict):
        raise ValueError("ANN candidates must be an object keyed by source token ID")
    ann_config = {
        "enabled": True,
        "kind": "external-shared-embedding-candidates",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "provenance": provenance,
    }
    return candidates, ann_config


def _real_inputs(args: argparse.Namespace):
    if not all((args.source, args.target, args.source_revision, args.target_revision)):
        raise ValueError(
            "real build requires --source/--target and both pinned revisions"
        )
    formal_values = (args.records_jsonl, args.manifest_json, args.build_split)
    if args.texts_jsonl and any(formal_values):
        raise ValueError("preview texts and manifest-bound corpus modes cannot mix")
    if args.texts_jsonl:
        corpus_mode = "preview"
    elif all(formal_values):
        corpus_mode = "manifest"
    else:
        raise ValueError(
            "real build requires --texts-jsonl or records/manifest/build-split"
        )
    from transformers import AutoTokenizer

    source = AutoTokenizer.from_pretrained(
        args.source, revision=args.source_revision, use_fast=True
    )
    target = AutoTokenizer.from_pretrained(
        args.target, revision=args.target_revision, use_fast=True
    )
    if corpus_mode == "manifest":
        texts, data_config = load_manifest_texts(
            args.records_jsonl,
            args.manifest_json,
            build_split=args.build_split,
        )
    else:
        texts = []
        for line_number, line in enumerate(
            args.texts_jsonl.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            text = value if isinstance(value, str) else value.get("text")
            if not isinstance(text, str):
                raise ValueError(f"line {line_number} must be a string or contain text")
            texts.append(text)
        data_config = {
            "mode": "direct-preview-texts",
            "texts_sha256": file_sha256(args.texts_jsonl),
            "canonical_messages": len(texts),
        }
    ann_fallback = None
    ann_config: dict[str, Any] = {"enabled": False}
    if args.ann_candidates_json:
        candidates, ann_config = _load_ann_candidates(args.ann_candidates_json)
        ann_fallback = lambda source_id, _: candidates.get(str(source_id), [])
    return source, target, texts, ann_fallback, ann_config, data_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--source-revision")
    parser.add_argument("--target-revision")
    parser.add_argument("--texts-jsonl", type=Path)
    parser.add_argument("--records-jsonl", type=Path)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--build-split", choices=("transport_train", "transport_dev"))
    parser.add_argument("--ann-candidates-json", type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--audit-markdown", required=True, type=Path)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-version")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    checkpoint = args.artifact.with_suffix(args.artifact.suffix + ".checkpoint.json")
    if args.resume and args.artifact.exists():
        report = audit_transport_artifact(load_transport_artifact(args.artifact))
        save_audit(report, args.audit_json, args.audit_markdown)
        _write_checkpoint(
            checkpoint, {"status": "complete", "resume": "loaded-valid-artifact"}
        )
        return 0
    prior_checkpoint = checkpoint.exists()
    _write_checkpoint(
        checkpoint,
        {
            "status": "building",
            "resume": "restart-from-recorded-inputs" if prior_checkpoint else "fresh",
        },
    )
    source, target, texts, ann_fallback, ann_config, data_config = (
        _toy_inputs() if args.toy else _real_inputs(args)
    )
    result = build_vocab_transport(
        source,
        target,
        texts,
        epsilon=args.epsilon,
        tolerance=args.tolerance,
        max_iter=args.max_iter,
        smoothing=args.smoothing,
        ann_fallback=ann_fallback,
        ann_config=ann_config,
        data_config=data_config,
        seed=args.seed,
        code_version=args.code_version or _git_version(),
    )
    partial = args.artifact.with_suffix(args.artifact.suffix + ".partial.npz")
    save_transport_artifact(result.artifact, partial)
    verified = load_transport_artifact(partial)
    report = audit_transport_artifact(verified)
    if not report["valid"]:
        raise ValueError("built artifact failed independent audit")
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(args.artifact)
    save_audit(report, args.audit_json, args.audit_markdown)
    _write_checkpoint(
        checkpoint,
        {
            "status": "complete",
            "resume": "restarted" if prior_checkpoint else "fresh",
            "artifact": str(args.artifact),
            "input_fingerprint": result.artifact.metadata["input_fingerprint"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
