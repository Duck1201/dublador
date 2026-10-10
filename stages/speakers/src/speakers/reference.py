"""Choosing a voice-cloning reference clip from the transcript (pure functions)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_SPEAKER = "SPK_00"


@dataclass(frozen=True)
class Candidate:
    speaker: str
    start: float
    end: float
    text: str
    word_score: float

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


def _mean_score(segments: Sequence[dict[str, Any]]) -> float:
    scores = [w["score"] for s in segments for w in s.get("words", []) if "score" in w]
    return float(np.mean(scores)) if scores else 0.5


def candidates(
    segments: Sequence[dict[str, Any]],
    *,
    min_seconds: float,
    max_seconds: float,
    max_gap: float,
) -> list[Candidate]:
    """Runs of consecutive same-speaker segments with short gaps, min..max seconds long."""
    found: list[Candidate] = []
    for i, first in enumerate(segments):
        speaker = speaker_of(first)
        run = [first]
        for nxt in segments[i + 1 :]:
            if speaker_of(nxt) != speaker or nxt["start"] - run[-1]["end"] > max_gap:
                break
            if nxt["end"] - first["start"] > max_seconds:
                break
            run.append(nxt)
        # Every prefix of the run that is long enough is a candidate.
        for j in range(len(run)):
            start, end = run[0]["start"], run[j]["end"]
            if min_seconds <= end - start <= max_seconds:
                found.append(
                    Candidate(
                        speaker=speaker,
                        start=start,
                        end=end,
                        text=" ".join(s["text"].strip() for s in run[: j + 1]),
                        word_score=_mean_score(run[: j + 1]),
                    )
                )
    return found


def _db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return 20 * np.log10(max(rms, 1e-9))


def score(cand: Candidate, vocals: np.ndarray, accompaniment: np.ndarray, sr: int) -> float:
    """Higher is better: clean voice (little music under it), ~10 s long, confident words."""
    a, b = int(cand.start * sr), int(cand.end * sr)
    voice_to_music = _db(vocals[a:b]) - _db(accompaniment[a:b])
    return voice_to_music - 0.5 * abs(cand.duration - 10.0) + 10.0 * cand.word_score


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
    max_gap: float,
) -> tuple[Candidate | None, str | None]:
    """Best clip for a speaker, and a warning when only a short one was found."""
    own = [s for s in segments if speaker_of(s) == speaker]
    for minimum, warning in (
        (min_seconds, None),
        (fallback_min_seconds, f"{speaker}: no clean {min_seconds:g}s clip, using a shorter one"),
    ):
        found = candidates(own, min_seconds=minimum, max_seconds=max_seconds, max_gap=max_gap)
        if found:
            best = max(found, key=lambda c: score(c, vocals, accompaniment, sr))
            return best, warning
    return None, f"{speaker}: no clip of {fallback_min_seconds:g}s or more; needs a bank voice"
