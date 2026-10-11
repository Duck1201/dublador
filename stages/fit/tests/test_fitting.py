"""Strategy decision for one segment."""

import math

import pytest
from fit.fitting import attempt, decide, room, shorten_to

KW = {"max_speedup": 1.25, "max_forced": 1.6, "min_speedup": 0.85}


def test_room_reaches_the_next_line_minus_the_gap() -> None:
    assert room(1.0, 2.0, next_start=4.15, gap=0.15) == pytest.approx(3.0)
    assert room(1.0, 2.0, next_start=2.1, gap=0.15) == pytest.approx(1.0)  # never below own
    assert room(1.0, 2.0, next_start=None, gap=0.15) == math.inf


@pytest.mark.parametrize(
    ("natural", "strategy", "ratio"),
    [
        (0.9, "fits", 0.9),  # slowed down to the original length
        (0.5, "fits", 0.85),  # never slower than min_speedup
        (1.0, "fits", 1.0),
        (2.5, "silence", 1.0),
        (3.6, "stretch", 1.2),
        (4.5, "overflow", 1.5),
        (9.0, "overflow", 1.6),  # forced speed-up is capped
    ],
)
def test_decide(natural: float, strategy: str, ratio: float) -> None:
    fit = decide(natural, own=1.0, available=3.0, **KW)
    assert fit.strategy == strategy and fit.ratio == pytest.approx(ratio)


def test_attempts_and_shorten_target() -> None:
    assert attempt(None, overflow=True) == 1
    assert attempt(1, overflow=True) == 2
    assert attempt(2, overflow=False) == 0
    assert shorten_to(20, natural=4.0, available=3.0) == 15
