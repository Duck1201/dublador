"""Pure post-processing of WhisperX output: hallucination filters and transcript/v1."""

import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

SAMPLE_RATE = 16000
_WORD = re.compile(r"[\w']+")


def rms_dbfs(audio: np.ndarray, start: float, end: float, sr: int = SAMPLE_RATE) -> float:
    """Loudness of audio[start:end] in dBFS (-inf for digital silence)."""
    chunk = audio[int(start * sr) : max(int(end * sr), int(start * sr) + 1)]
    if chunk.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
    return 20 * np.log10(rms) if rms > 0 else float("-inf")


def is_repetitive(text: str, *, max_repeats: int = 4) -> bool:
    """A word or short phrase looping many times in a row (classic Whisper failure)."""
    words = [w.lower() for w in _WORD.findall(text)]
    for n in (1, 2, 3, 4):
        for i in range(len(words) - n * max_repeats + 1):
            gram = words[i : i + n]
            if all(words[i + k * n : i + (k + 1) * n] == gram for k in range(1, max_repeats)):
                return True
    return False


def drop_reason(
    segment: dict[str, Any],
    previous_texts: Sequence[str],
    audio: np.ndarray,
    *,
    silence_dbfs: float,
) -> str | None:
    """Why a segment looks like a hallucination, or None to keep it."""
    text = segment["text"].strip()
    if not _WORD.search(text):
        return "no words"
    if rms_dbfs(audio, segment["start"], segment["end"]) < silence_dbfs:
        return "text over silence"
    if is_repetitive(text):
        return "repetition loop"
    if len(previous_texts) >= 2 and all(p.strip() == text for p in previous_texts[-2:]):
        return "same text three times in a row"
    return None


def _timed(word: dict[str, Any]) -> bool:
    return isinstance(word.get("start"), int | float) and isinstance(word.get("end"), int | float)


def _piece(words: list[dict[str, Any]], fallback: dict[str, Any]) -> dict[str, Any]:
    timed = [w for w in words if _timed(w)]
    return {
        "start": timed[0]["start"] if timed else fallback["start"],
        "end": timed[-1]["end"] if timed else fallback["end"],
        "text": " ".join(w["word"].strip() for w in words),
        "words": words,
    }


def split_segment(
    segment: dict[str, Any], *, max_duration: float, min_pause: float
) -> list[dict[str, Any]]:
    """Split an ASR segment into sentence-sized pieces using word timestamps.

    Batched Whisper returns VAD chunks of up to 30 s that can span many
    sentences. Cut after sentence punctuation and at pauses >= min_pause;
    pieces still longer than max_duration are cut at their widest gap.
    """
    words = [w for w in segment.get("words", []) if w.get("word", "").strip()]
    if len(words) < 2 or not all(_timed(w) for w in (words[0], words[-1])):
        return [segment]

    pieces: list[list[dict[str, Any]]] = [[]]
    for i, word in enumerate(words):
        pieces[-1].append(word)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None:
            break
        sentence_end = word["word"].rstrip().endswith((".", "?", "!"))
        gap = nxt["start"] - word["end"] if _timed(word) and _timed(nxt) else 0.0
        if sentence_end or gap >= min_pause:
            pieces.append([])

    result: list[list[dict[str, Any]]] = []
    for piece in pieces:
        result.extend(_cut_long(piece, max_duration))
    return [_piece(p, segment) for p in result if p]


def _cut_long(words: list[dict[str, Any]], max_duration: float) -> list[list[dict[str, Any]]]:
    timed = [w for w in words if _timed(w)]
    if len(words) < 2 or not timed or timed[-1]["end"] - timed[0]["start"] <= max_duration:
        return [words]
    gaps = [
        (words[i + 1]["start"] - words[i]["end"], i)
        for i in range(len(words) - 1)
        if _timed(words[i]) and _timed(words[i + 1])
    ]
    if not gaps:
        return [words]
    _, cut = max(gaps)
    return _cut_long(words[: cut + 1], max_duration) + _cut_long(words[cut + 1 :], max_duration)


def _word(raw: dict[str, Any]) -> dict[str, Any]:
    word: dict[str, Any] = {"word": raw["word"].strip()}
    for key in ("start", "end"):
        if isinstance(raw.get(key), int | float):
            word[key] = round(float(raw[key]), 3)
    if isinstance(raw.get("score"), int | float):
        word["score"] = round(float(raw["score"]), 3)
    return word


def build_transcript(
    segments: Iterable[dict[str, Any]],
    audio: np.ndarray,
    *,
    language: str,
    silence_dbfs: float,
    max_duration: float = 12.0,
    min_pause: float = 0.6,
) -> tuple[dict[str, Any], list[str]]:
    """transcript/v1 document with stable ids, plus warnings for dropped segments."""
    kept: list[dict[str, Any]] = []
    warnings: list[str] = []
    texts: list[str] = []
    pieces = [
        piece
        for raw in segments
        for piece in split_segment(raw, max_duration=max_duration, min_pause=min_pause)
    ]
    for raw in pieces:
        if raw.get("start") is None or raw.get("end") is None:
            continue
        reason = drop_reason(raw, texts, audio, silence_dbfs=silence_dbfs)
        texts.append(raw["text"])
        if reason:
            span = f"{raw['start']:.2f}-{raw['end']:.2f}s"
            warnings.append(f"dropped {span} ({reason}): {raw['text'].strip()[:60]}")
            continue
        start, end = round(float(raw["start"]), 3), round(float(raw["end"]), 3)
        kept.append(
            {
                "start": start,
                "end": max(start, end),
                "text": raw["text"].strip(),
                "speaker": None,
                "words": [_word(w) for w in raw.get("words", []) if w.get("word", "").strip()],
            }
        )
    for i, seg in enumerate(kept, start=1):
        seg["id"] = f"seg_{i:04d}"
    ordered = [{"id": s.pop("id"), **s} for s in kept]
    return {"schema": "transcript/v1", "language": language, "segments": ordered}, warnings
