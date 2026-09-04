"""Tokenizer-independent raw-byte, offset, and special-token metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def _bytes_to_unicode() -> Dict[int, str]:
    safe = list(range(ord("!"), ord("~") + 1))
    safe += list(range(ord("¡"), ord("¬") + 1))
    safe += list(range(ord("®"), ord("ÿ") + 1))
    chars = safe[:]
    extra = 0
    for byte in range(256):
        if byte not in safe:
            safe.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(safe, (chr(value) for value in chars)))


_UNICODE_TO_BYTE = {value: key for key, value in _bytes_to_unicode().items()}
_CONTROL_ID_ATTRIBUTES = (
    ("bos_token_id", "bos"),
    ("eos_token_id", "eos"),
    ("pad_token_id", "pad"),
    ("unk_token_id", "unk"),
    ("sep_token_id", "sep"),
    ("cls_token_id", "cls"),
    ("mask_token_id", "mask"),
)


@dataclass(frozen=True)
class TokenMetadata:
    token_id: int
    token: str
    raw_bytes: bytes
    is_special: bool
    special_kind: str | None


def token_raw_bytes(tokenizer: Any, token_id: int) -> bytes:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        raise ValueError(f"Tokenizer returned no token for id {token_id}")
    token = str(token)
    if all(char in _UNICODE_TO_BYTE for char in token):
        return bytes(_UNICODE_TO_BYTE[char] for char in token)
    return token.encode("utf-8")


def special_id_to_token(tokenizer: Any) -> Dict[int, str]:
    vocabulary = tokenizer.get_vocab()
    result: Dict[int, str] = {}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder_getter = getattr(backend, "get_added_tokens_decoder", None)
    if callable(decoder_getter):
        for token_id, token in decoder_getter().items():
            if bool(getattr(token, "special", False)):
                result[int(token_id)] = str(token)
    for token in getattr(tokenizer, "all_special_tokens", []):
        token_id = vocabulary.get(token)
        if token_id is not None:
            result[int(token_id)] = str(token)
    for attribute, _ in _CONTROL_ID_ATTRIBUTES:
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            if token is not None:
                result[int(token_id)] = str(token)
    return result


def special_id_to_kind(tokenizer: Any) -> Dict[int, str]:
    result = {token_id: "special" for token_id in special_id_to_token(tokenizer)}
    for attribute, kind in _CONTROL_ID_ATTRIBUTES:
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            result[int(token_id)] = kind
    return result


def iter_token_metadata(tokenizer: Any) -> Iterable[TokenMetadata]:
    kinds = special_id_to_kind(tokenizer)
    for token, token_id in sorted(
        tokenizer.get_vocab().items(), key=lambda item: item[1]
    ):
        token_id = int(token_id)
        yield TokenMetadata(
            token_id=token_id,
            token=str(token),
            raw_bytes=token_raw_bytes(tokenizer, token_id),
            is_special=token_id in kinds,
            special_kind=kinds.get(token_id),
        )


def ordinary_bytes_index(tokenizer: Any) -> Dict[bytes, List[int]]:
    result: Dict[bytes, List[int]] = {}
    for metadata in iter_token_metadata(tokenizer):
        if metadata.is_special:
            continue
        result.setdefault(metadata.raw_bytes, []).append(metadata.token_id)
    return result


def exact_byte_matches(
    source_tokenizer: Any, target_tokenizer: Any
) -> Dict[int, List[int]]:
    target_index = ordinary_bytes_index(target_tokenizer)
    result: Dict[int, List[int]] = {}
    for metadata in iter_token_metadata(source_tokenizer):
        if not metadata.is_special and metadata.raw_bytes in target_index:
            result[metadata.token_id] = list(target_index[metadata.raw_bytes])
    return result


def encode_with_byte_spans(tokenizer: Any, text: str) -> List[Tuple[int, int, int]]:
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("tokenizer must be fast and provide offset mappings")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]
    byte_prefix = [0]
    for character in text:
        byte_prefix.append(byte_prefix[-1] + len(character.encode("utf-8")))
    spans: List[Tuple[int, int, int]] = []
    for token_id, (character_start, character_end) in zip(ids, offsets):
        if not 0 <= character_start <= character_end <= len(text):
            raise ValueError(
                f"invalid character offset {(character_start, character_end)} "
                f"for text length {len(text)}"
            )
        if character_start != character_end:
            spans.append(
                (
                    int(token_id),
                    byte_prefix[character_start],
                    byte_prefix[character_end],
                )
            )
    return spans


def tokenizer_fingerprint(tokenizer: Any) -> str:
    payload: Mapping[str, Any] = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
        "vocab": sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]),
        "special_tokens": sorted(special_id_to_token(tokenizer).items()),
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
