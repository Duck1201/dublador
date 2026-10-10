"""Windowed, validated translation of transcript segments (no I/O)."""

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from stagekit import hash_value

PROMPT_VERSION = "v0.3"
CHARS_PER_SECOND = 15  # rough pt-BR speaking rate; calibrated per TTS in M3

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["translations"],
}

SYSTEM_PROMPT = """You translate {source} dialogue into {target} for dubbing.
Rules:
- Write natural, spoken {target} (Brazilian Portuguese: use "você",
  Brazilian vocabulary and spelling).
- Keep the meaning, tone and register of each line. Do not add explanations or notes.
- Use normal capitalization and punctuation, even when the source line lacks them.
- Keep names, brands and technical terms that people say in English.
- Each line is dubbed over the original timing: keep it about as short as the original,
  never longer than its "max_chars".
- Translate only the lines in "translate"; the context lines are for reference.
- Answer with JSON: {{"translations": [{{"id": "...", "text": "..."}}]}},
  one entry per id, in the same order."""

Chat = Callable[[list[dict[str, str]]], str]


class TranslationError(Exception):
    pass


@dataclass(frozen=True)
class Segment:
    id: str
    start: float
    end: float
    speaker: str
    text: str


def input_hash(seg: Segment, *, model: str, source: str, target: str) -> str:
    return hash_value(
        {
            "text": seg.text,
            "model": model,
            "source": source,
            "target": target,
            "prompt": PROMPT_VERSION,
        }
    )


def _line(seg: Segment) -> dict[str, Any]:
    seconds = round(seg.end - seg.start, 1)
    return {
        "id": seg.id,
        "speaker": seg.speaker,
        "seconds": seconds,
        "max_chars": max(8, int(seconds * CHARS_PER_SECOND)),
        "text": seg.text,
    }


def build_messages(
    batch: Sequence[Segment],
    before: Sequence[tuple[Segment, str | None]],
    after: Sequence[Segment],
    *,
    source: str,
    target: str,
) -> list[dict[str, str]]:
    payload = {
        "context_before": [
            {"speaker": s.speaker, "text": s.text, **({"translation": t} if t else {})}
            for s, t in before
        ],
        "translate": [_line(s) for s in batch],
        "context_after": [{"speaker": s.speaker, "text": s.text} for s in after],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(source=source, target=target)},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


_SENTENCE_START = re.compile(r"(^|[.!?…]\s+)([\"'«“(¿¡]*)(\w)")


def tidy(text: str) -> str:
    """Capitalize the start of each sentence; small models often answer in lowercase."""
    return _SENTENCE_START.sub(lambda m: m.group(1) + m.group(2) + m.group(3).upper(), text.strip())


def parse_response(raw: str, expected: Sequence[str]) -> dict[str, str]:
    """Translations by id, or a TranslationError saying what is wrong."""
    try:
        data = json.loads(raw)
        items = data["translations"]
        found = {str(item["id"]): tidy(str(item["text"])) for item in items}
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise TranslationError(f"response is not the expected JSON ({e})") from e
    missing = [i for i in expected if i not in found]
    if missing:
        raise TranslationError(f"missing ids: {', '.join(missing)}")
    empty = [i for i in expected if not found[i]]
    if empty:
        raise TranslationError(f"empty translations for: {', '.join(empty)}")
    return {i: found[i] for i in expected}


def translate_batch(
    chat: Chat,
    batch: Sequence[Segment],
    before: Sequence[tuple[Segment, str | None]],
    after: Sequence[Segment],
    *,
    source: str,
    target: str,
    retries: int,
) -> dict[str, str]:
    """Ask, validate, re-ask with the error; split the batch if it keeps failing."""
    messages = build_messages(batch, before, after, source=source, target=target)
    expected = [s.id for s in batch]
    error: TranslationError | None = None
    for _ in range(retries + 1):
        raw = chat(messages)
        try:
            return parse_response(raw, expected)
        except TranslationError as e:
            error = e
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Invalid answer: {e}. Answer again with every id."},
            ]
    if len(batch) == 1:
        raise TranslationError(f"{batch[0].id}: {error}")
    half = len(batch) // 2
    first = translate_batch(
        chat, batch[:half], before, batch[half:], source=source, target=target, retries=retries
    )
    tail_context = [*before, *((s, first[s.id]) for s in batch[:half])]
    second = translate_batch(
        chat, batch[half:], tail_context, after, source=source, target=target, retries=retries
    )
    return {**first, **second}


def translate_all(
    chat: Chat,
    segments: Sequence[Segment],
    *,
    todo: set[str],
    known: dict[str, str],
    source: str,
    target: str,
    batch_size: int,
    context: int,
    retries: int,
    on_progress: Callable[[int, int], None] = lambda done, total: None,
) -> dict[str, str]:
    """Translate the segments in `todo`; `known` holds reusable translations."""
    result = dict(known)
    pending = [s for s in segments if s.id in todo]
    index = {s.id: i for i, s in enumerate(segments)}
    done = 0
    for b in range(0, len(pending), batch_size):
        batch = pending[b : b + batch_size]
        first, last = index[batch[0].id], index[batch[-1].id]
        before = [(s, result.get(s.id)) for s in segments[max(0, first - context) : first]]
        after = segments[last + 1 : last + 1 + context]
        result.update(
            translate_batch(
                chat, batch, before, after, source=source, target=target, retries=retries
            )
        )
        done += len(batch)
        on_progress(done, len(pending))
    return result
