"""Edge-silence trimming of synthesized segments."""

import numpy as np
from synthesize.audio import trim_silence

SR = 1000


def tone(seconds: float) -> np.ndarray:
    return 0.5 * np.sin(np.arange(int(seconds * SR)) * 0.3).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_trims_edges_but_keeps_inner_pauses_and_padding() -> None:
    audio = np.concatenate([silence(1.0), tone(0.5), silence(0.3), tone(0.5), silence(2.0)])
    out = trim_silence(audio, SR)
    assert len(out) == int((0.5 + 0.3 + 0.5 + 2 * 0.05) * SR)


def test_silent_or_tiny_audio_is_left_alone() -> None:
    assert len(trim_silence(silence(1.0), SR)) == SR
    assert len(trim_silence(np.zeros(3, dtype=np.float32), SR)) == 3
