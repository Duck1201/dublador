"""Which segments need synthesis (per-segment cache), without touching the model."""

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stagekit import hash_file, hash_value

# Generation budget per segment. Normal pt-BR speech runs ~0.07 s per character; the
# budget leaves room for slow voices while stopping a runaway generation early.
MIN_SECONDS = 4.0
WINDOW_FACTOR = 3.0
SECONDS_PER_CHAR = 0.2


@dataclass(frozen=True)
class Voice:
    speaker: str
    ref_wav: Path
    ref_text: str
    ref_hash: str


@dataclass(frozen=True)
class Job:
    id: str
    speaker: str
    text: str
    target_duration: float
    input_hash: str


def voices(speakers: Mapping[str, Any], job_dir: Path) -> dict[str, Voice]:
    """Cloning voices by speaker id; speakers without a reference are left out."""
    found = {}
    for spk in speakers["speakers"]:
        ref = spk.get("reference")
        if spk["strategy"] == "clone" and ref:
            wav = job_dir / ref["path"]
            found[spk["id"]] = Voice(spk["id"], wav, ref["text"], hash_file(wav))
    return found


def jobs(
    translation: Mapping[str, Any],
    voices_by_speaker: Mapping[str, Voice],
    backend: Mapping[str, Any],
) -> list[Job]:
    result = []
    for seg in translation["segments"]:
        voice = voices_by_speaker.get(seg["speaker"])
        key = {
            "text": seg["text"],
            "voice": [voice.ref_hash, voice.ref_text] if voice else None,
            "backend": dict(backend),
        }
        result.append(
            Job(
                seg["id"],
                seg["speaker"],
                seg["text"],
                round(seg["end"] - seg["start"], 3),
                hash_value(key),
            )
        )
    return result


def max_seconds(job: Job) -> float:
    """The most audio a segment may need; a backend that hits it has run away."""
    return max(MIN_SECONDS, WINDOW_FACTOR * job.target_duration, SECONDS_PER_CHAR * len(job.text))


def ran_away(duration: float, budget: float) -> bool:
    return duration >= budget - 0.1  # backends stop on a token boundary just under the budget


def split(
    all_jobs: Sequence[Job],
    previous: Mapping[str, Mapping[str, Any]],
    existing: Container[str],
    only: Container[str] | None,
) -> tuple[list[Job], list[Job]]:
    """(to synthesize, reusable). Reuse needs the same input hash and the WAV still on disk;
    --only forces its ids."""
    todo, keep = [], []
    for job in all_jobs:
        cached = previous.get(job.id)
        reusable = (
            cached is not None and cached["input_hash"] == job.input_hash and job.id in existing
        )
        forced = only is not None and job.id in only
        (keep if reusable and not forced else todo).append(job)
    return todo, keep
