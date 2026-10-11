"""How each dubbed line fits its timing window (pure; ADR-0004, ADR-0018).

Per segment, in order:
1. fits: the clip is no longer than the original line;
2. silence: it runs into the pause before the next line (of any speaker), up to
   the next line's start minus a gap;
3. stretch: sped up (pitch kept) to that room, up to `max_speedup`;
4. overflow: still too long. The clip is sped up past the limit (capped) so the
   mix always has audio, and the fit loop is asked to shorten the translation
   (first overflow) and then to re-synthesize at the target duration.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Fit:
    strategy: str  # fits | silence | stretch | overflow
    ratio: float  # speed factor applied (1 = untouched)
    room: float  # seconds available until the next line minus the gap


def room(start: float, end: float, next_start: float | None, gap: float) -> float:
    """Seconds this line may take: up to the next line's start minus `gap`, and
    never less than the original line. The last line is not limited."""
    if next_start is None:
        return math.inf
    return max(next_start - gap - start, end - start)


def decide(
    natural: float, own: float, available: float, *, max_speedup: float, max_forced: float
) -> Fit:
    if natural <= own:
        return Fit("fits", 1.0, available)
    if natural <= available:
        return Fit("silence", 1.0, available)
    ratio = natural / available
    if ratio <= max_speedup:
        return Fit("stretch", ratio, available)
    return Fit("overflow", min(ratio, max_forced), available)


def attempt(previous_attempt: int | None, overflow: bool) -> int:
    """Overflows in a row for a segment across fit-loop iterations."""
    return (previous_attempt or 0) + 1 if overflow else 0


def shorten_to(syllables: int, natural: float, available: float) -> int:
    """Syllables that would fit the room at the speech rate of the current clip."""
    return max(1, math.floor(syllables * available / natural))
