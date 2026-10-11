"""The pt-BR fallback voice bank (ADR-0002, ADR-0017; voices/README.md).

Voices are designed from a text description by the TTS backend once, at
`dublador setup`, and kept in <models_dir>/voices/pt-BR/<gender>/NN.wav with
the spoken text in NN.txt. They are then cloned like any speaker reference.
Keep this table in sync with voices/README.md.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

# id -> (description for the backend's voice design, seed), from low to high
# pitch within each gender; the speakers stage measures the WAVs to match a
# speaker's pitch. No "very low pitch" designs: they sounded robotic in the
# reference-video A/B test (ADR-0021). Median F0s are in voices/README.md.
VOICES: dict[str, tuple[str, int]] = {
    "male_01": ("male, young adult, low pitch", 12),
    "male_02": ("male, middle-aged, low pitch", 102),
    "male_03": ("male, young adult, moderate pitch", 101),
    "male_04": ("male, middle-aged, moderate pitch", 104),
    "female_01": ("female, middle-aged, low pitch", 204),
    "female_02": ("female, young adult, low pitch", 12),
    "female_03": ("female, middle-aged, moderate pitch", 202),
    "female_04": ("female, young adult, moderate pitch", 201),
    "female_05": ("female, young adult, high pitch", 203),
}
# Short and neutral, like the 3-4.5 s cloning references (ADR-0017).
TEXT = "Olá, tudo bem? Hoje eu vou explicar uma ideia simples, passo a passo."


def bank_dir(models_dir: Path) -> Path:
    return models_dir / "voices" / "pt-BR"


def paths(models_dir: Path, voice: str) -> tuple[Path, Path]:
    gender, n = voice.rsplit("_", 1)
    folder = bank_dir(models_dir) / gender
    return folder / f"{n}.wav", folder / f"{n}.txt"


def missing(models_dir: Path) -> list[str]:
    return [v for v in VOICES if not all(p.is_file() for p in paths(models_dir, v))]


def build(
    models_dir: Path,
    design: Callable[[str, str, int], tuple[np.ndarray, int]],
    log: Callable[[str], None] = print,
) -> None:
    """Design every voice that is not on disk yet; `design(description, text, seed)`."""
    import soundfile

    for voice in missing(models_dir):
        description, seed = VOICES[voice]
        audio, sr = design(description, TEXT, seed)
        wav, txt = paths(models_dir, voice)
        wav.parent.mkdir(parents=True, exist_ok=True)
        tmp = wav.with_name(f".{wav.name}.tmp.wav")
        soundfile.write(str(tmp), audio, sr, subtype="PCM_16")
        tmp.replace(wav)
        txt.write_text(TEXT + "\n")
        log(f"voice bank: {voice} ({description})")
