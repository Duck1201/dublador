"""Voice-bank choice (ADR-0002, ADR-0020; voices/README.md).

The bank (one WAV + text per voice, designed by the synthesize stage at
`dublador setup`) lives in <models_dir>/voices/pt-BR/<gender>/NN.wav. This
stage only reads the WAVs to measure their pitch and picks an id per speaker.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

GENDERS = ("male", "female")


@dataclass(frozen=True)
class BankVoice:
    id: str  # "<gender>_NN"
    gender: str
    f0: float | None


def scan(models_dir: Path, pitch: Callable[[Path], float | None]) -> list[BankVoice]:
    """Bank voices on disk with their median pitch, in id order."""
    root = models_dir / "voices" / "pt-BR"
    voices = []
    for gender in GENDERS:
        for wav in sorted((root / gender).glob("[0-9][0-9].wav")):
            if wav.with_suffix(".txt").is_file():
                voices.append(BankVoice(f"{gender}_{wav.stem}", gender, pitch(wav)))
    return voices


def _distance(a: float | None, b: float | None) -> float:
    if a is None or b is None:
        return math.inf
    return abs(math.log(a / b))  # semitone-like: 100 vs 120 Hz ~ 200 vs 240 Hz


def assign(
    speakers: Sequence[tuple[str, str, float | None]], voices: Sequence[BankVoice]
) -> dict[str, str]:
    """speakers: (id, gender, median F0), most speech first. Each gets the unused
    voice of its gender (any gender if `unknown`) closest in pitch; without a
    pitch, the first unused one. Voices repeat only when they run out."""
    used: set[str] = set()
    chosen: dict[str, str] = {}
    for speaker, gender, f0 in speakers:
        pool = [v for v in voices if v.gender == gender] or list(voices)
        if not pool:
            raise ValueError("the voice bank is empty")
        free = [v for v in pool if v.id not in used] or pool
        best = min(free, key=lambda v: (_distance(f0, v.f0), v.id))
        chosen[speaker] = best.id
        used.add(best.id)
    return chosen
