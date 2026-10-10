"""Voice-bank choice for speakers without a usable reference (ADR-0002).

The bank itself (WAV + text per voice) lives in models_dir and is built by the
synthesize stage; this stage only picks ids, so it never touches models_dir.
"""

from collections.abc import Sequence

GENDERS = ("male", "female")


def voice_id(gender: str, n: int) -> str:
    return f"{gender}_{n:02d}"


def assign(speakers: Sequence[tuple[str, str]], *, per_gender: int) -> dict[str, str]:
    """speakers: (id, gender) of each speaker that needs a bank voice, most speech
    first. Each gets the next unused voice of its gender; `unknown` alternates
    genders. Voices repeat only when a gender runs out."""
    used = dict.fromkeys(GENDERS, 0)
    chosen: dict[str, str] = {}
    unknown = 0
    for speaker, gender in speakers:
        if gender not in GENDERS:
            gender = GENDERS[unknown % len(GENDERS)]
            unknown += 1
        chosen[speaker] = voice_id(gender, used[gender] % per_gender + 1)
        used[gender] += 1
    return chosen
