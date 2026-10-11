"""Choosing a voice-cloning reference clip from the transcript (pure functions)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_SPEAKER = "SPK_00"
TAIL = 0.08  # seconds kept after the last word of a reference
# A reference that starts or ends mid-phrase leaks its edge sounds into the cloned
# speech (an English "...they" became a spoken "Ei" in pt-BR): prefer clips that
# start after and end before a sentence end or a pause this long.
EDGE_PAUSE = 0.25


@dataclass(frozen=True)
class Candidate:
    speaker: str
    start: float
    end: float
    text: str
    word_score: float
    clean_edges: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


def speaker_of(segment: dict[str, Any]) -> str:
    return segment.get("speaker") or DEFAULT_SPEAKER


def speech_seconds(segments: Sequence[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for seg in segments:
        spk = speaker_of(seg)
        totals[spk] = totals.get(spk, 0.0) + seg["end"] - seg["start"]
    return totals


def _words(segment: dict[str, Any]) -> list[dict[str, Any]]:
    """Timed words of a segment; a segment without them counts as one word."""
    timed = [w for w in segment.get("words", []) if "start" in w and "end" in w]
    if timed:
        return timed
    return [{"word": segment["text"], "start": segment["start"], "end": segment["end"]}]


def candidates(
    segments: Sequence[dict[str, Any]],
    *,
    min_seconds: float,
    max_seconds: float,
    max_gap: float,
) -> list[Candidate]:
    """Word-aligned clips min..max seconds long inside runs of consecutive same-speaker
    segments with short gaps. A clip never cuts a word."""
    found: list[Candidate] = []
    i = 0
    while i < len(segments):
        speaker = speaker_of(segments[i])
        run = [segments[i]]
        while (
            i + len(run) < len(segments)
            and speaker_of(segments[i + len(run)]) == speaker
            and segments[i + len(run)]["start"] - run[-1]["end"] <= max_gap
        ):
            run.append(segments[i + len(run)])
        words = [w for seg in run for w in _words(seg)]
        for a, first in enumerate(words):
            for b in range(a, len(words)):
                last = words[b]
                if last["end"] - first["start"] > max_seconds:
                    break
                if last["end"] - first["start"] < min_seconds:
                    continue
                # A little tail keeps the last phoneme, without reaching the next word.
                nxt = words[b + 1]["start"] if b + 1 < len(words) else last["end"] + TAIL
                chosen = words[a : b + 1]
                scores = [w["score"] for w in chosen if "score" in w]
                clean_start = a == 0 or _boundary(words[a - 1], first)
                clean_end = b + 1 == len(words) or _boundary(last, words[b + 1])
                found.append(
                    Candidate(
                        speaker=speaker,
                        start=first["start"],
                        end=min(last["end"] + TAIL, max(nxt, last["end"])),
                        text=" ".join(w["word"].strip() for w in chosen),
                        word_score=float(np.mean(scores)) if scores else 0.5,
                        clean_edges=clean_start and clean_end,
                    )
                )
        i += len(run)
    return found


def _boundary(before: dict[str, Any], after: dict[str, Any]) -> bool:
    """A sentence end or a pause between two consecutive words."""
    if before["word"].rstrip().endswith((".", "!", "?", ";", ":")):
        return True
    return after["start"] - before["end"] >= EDGE_PAUSE


def _db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20 * np.log10(max(rms, 1e-9))


def score(
    cand: Candidate, vocals: np.ndarray, accompaniment: np.ndarray, sr: int, target: float
) -> float:
    """Higher is better: clean voice (little music under it), close to `target` seconds,
    confident words."""
    a, b = int(cand.start * sr), int(cand.end * sr)
    voice_to_music = _db(vocals[a:b]) - _db(accompaniment[a:b])
    return voice_to_music - 2.0 * abs(cand.duration - target) + 10.0 * cand.word_score


def best_reference(
    segments: Sequence[dict[str, Any]],
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    sr: int,
    *,
    speaker: str,
    min_seconds: float,
    fallback_min_seconds: float,
    max_seconds: float,
    target_seconds: float,
    max_gap: float,
) -> tuple[Candidate | None, str | None]:
    """Best clip for a speaker, and a warning when only a short one was found."""
    own = [s for s in segments if speaker_of(s) == speaker]
    for minimum, warning in (
        (min_seconds, None),
        (fallback_min_seconds, f"{speaker}: no clean {min_seconds:g}s clip, using a shorter one"),
    ):
        found = candidates(own, min_seconds=minimum, max_seconds=max_seconds, max_gap=max_gap)
        found = [c for c in found if c.clean_edges] or found
        if found:
            best = max(found, key=lambda c: score(c, vocals, accompaniment, sr, target_seconds))
            return best, warning
    return None, f"{speaker}: no clip of {fallback_min_seconds:g}s or more; needs a bank voice"


def clean_clips(
    segments: Sequence[dict[str, Any]],
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    sr: int,
    *,
    speaker: str,
    count: int,
    min_seconds: float,
    max_seconds: float,
    max_gap: float,
) -> list[Candidate]:
    """Up to `count` non-overlapping clean clips of a speaker, best first (for gender)."""
    own = [s for s in segments if speaker_of(s) == speaker]
    found = candidates(own, min_seconds=min_seconds, max_seconds=max_seconds, max_gap=max_gap)
    target = (min_seconds + max_seconds) / 2
    ranked = sorted(found, key=lambda c: score(c, vocals, accompaniment, sr, target), reverse=True)
    chosen: list[Candidate] = []
    for cand in ranked:
        if all(cand.end <= c.start or cand.start >= c.end for c in chosen):
            chosen.append(cand)
            if len(chosen) == count:
                break
    return chosen
