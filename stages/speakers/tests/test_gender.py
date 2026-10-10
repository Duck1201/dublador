"""Gender vote and pitch, without the network."""

import numpy as np
from speakers.gender import median_f0, resample, vote


def tone(f0: float, seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    # A few harmonics, like a voice.
    return sum(np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3)).astype(np.float32)


def test_vote_weights_by_duration() -> None:
    # 9 s confidently female against 1 s confidently male.
    verdict = vote([(9.0, 0.95), (1.0, 0.05)], None, min_confidence=0.3)
    assert verdict.gender == "female" and verdict.confidence == 0.72  # |2 * 0.86 - 1|


def test_low_confidence_falls_back_to_clear_pitch() -> None:
    assert vote([(3.0, 0.55)], 120.0, min_confidence=0.3).gender == "male"
    assert vote([(3.0, 0.45)], 220.0, min_confidence=0.3).gender == "female"
    ambiguous = vote([(3.0, 0.5)], 165.0, min_confidence=0.3)
    assert ambiguous.gender == "unknown" and ambiguous.confidence is None
    assert vote([], None, min_confidence=0.3).gender == "unknown"


def test_median_f0_on_tones() -> None:
    assert abs((median_f0(tone(110.0), 16000) or 0) - 110.0) < 4
    assert abs((median_f0(tone(220.0), 16000) or 0) - 220.0) < 8
    assert median_f0(np.zeros(16000, dtype=np.float32), 16000) is None


def test_resample_to_model_rate() -> None:
    assert len(resample(np.zeros(44100, dtype=np.float32), 44100)) == 16000
