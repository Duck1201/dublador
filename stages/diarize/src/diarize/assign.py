"""Pure helpers: clean up diarization turns and give every transcript word a speaker."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def totals(turns: Sequence[Turn]) -> dict[str, float]:
    out: dict[str, float] = {}
    for t in turns:
        out[t.speaker] = out.get(t.speaker, 0.0) + t.duration
    return out


def merge_small_speakers(turns: Sequence[Turn], min_seconds: float) -> list[Turn]:
    """Give the turns of speakers with less than `min_seconds` of speech to the
    closest turn (in time) of a bigger speaker. Keeps everything when no speaker
    is big enough."""
    seconds = totals(turns)
    big = [t for t in turns if seconds[t.speaker] >= min_seconds]
    if not big:
        return list(turns)
    out = []
    for t in turns:
        if seconds[t.speaker] >= min_seconds:
            out.append(t)
        else:
            nearest = min(big, key=lambda b: _gap(t, b))
            out.append(replace(t, speaker=nearest.speaker))
    return out


def relabel(turns: Sequence[Turn]) -> list[Turn]:
    """SPK_00, SPK_01, … in order of first appearance."""
    names: dict[str, str] = {}
    for t in sorted(turns, key=lambda t: t.start):
        names.setdefault(t.speaker, f"SPK_{len(names):02d}")
    return [replace(t, speaker=names[t.speaker]) for t in turns]


def _gap(a: Turn | tuple[float, float], b: Turn) -> float:
    start, end = (a.start, a.end) if isinstance(a, Turn) else a
    return max(0.0, b.start - end, start - b.end)


def _overlap(start: float, end: float, t: Turn) -> float:
    return max(0.0, min(end, t.end) - max(start, t.start))


def speaker_at(start: float, end: float, turns: Sequence[Turn]) -> str | None:
    """The turn that covers [start, end] the most, or the nearest one."""
    if not turns:
        return None
    best = max(turns, key=lambda t: (_overlap(start, end, t), -_gap((start, end), t)))
    if _overlap(start, end, best) > 0:
        return best.speaker
    return min(turns, key=lambda t: _gap((start, end), t)).speaker


def _timed(word: dict[str, Any]) -> bool:
    return isinstance(word.get("start"), int | float) and isinstance(word.get("end"), int | float)


def split_by_speaker(
    segments: Sequence[dict[str, Any]], turns: Sequence[Turn]
) -> list[dict[str, Any]]:
    """transcript/v1 segments with `speaker` set, cut where the speaker changes
    between words; ids are renumbered seg_0001…"""
    out: list[dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words", [])
        if not any(_timed(w) for w in words):
            out.append({**seg, "speaker": speaker_at(seg["start"], seg["end"], turns)})
            continue
        pieces: list[tuple[str | None, list[dict[str, Any]]]] = []
        current = speaker_at(seg["start"], seg["end"], turns)
        for w in words:
            if _timed(w):
                current = speaker_at(w["start"], w["end"], turns)
            if pieces and pieces[-1][0] == current:
                pieces[-1][1].append(w)
            else:
                pieces.append((current, [w]))
        for speaker, ws in pieces:
            timed = [w for w in ws if _timed(w)]
            out.append(
                {
                    "id": "",
                    "start": timed[0]["start"] if timed else seg["start"],
                    "end": timed[-1]["end"] if timed else seg["end"],
                    "text": " ".join(w["word"] for w in ws),
                    "speaker": speaker,
                    "words": ws,
                }
            )
    for i, seg in enumerate(out, start=1):
        seg["id"] = f"seg_{i:04d}"
    return [{"id": s["id"], **{k: v for k, v in s.items() if k != "id"}} for s in out]
