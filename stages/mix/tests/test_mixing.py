import numpy as np
import pytest
from mix.mixing import fade, limit_peak, place, resample


def test_place_puts_samples_at_start_times() -> None:
    track = place([(0.5, np.ones(2, dtype=np.float32)), (1.0, np.full(2, 2, dtype=np.float32))],
                  rate=4, length=8, channels=2)  # fmt: skip
    assert track.shape == (8, 2)
    assert track[:, 0].tolist() == [0, 0, 1, 1, 2, 2, 0, 0]
    assert track[:, 1].tolist() == track[:, 0].tolist()


def test_place_overlapping_clips_add_and_track_grows() -> None:
    track = place([(0.0, np.ones(4, dtype=np.float32)), (0.5, np.ones(6, dtype=np.float32))],
                  rate=4, length=4, channels=1)  # fmt: skip
    assert track[:, 0].tolist() == [1, 1, 2, 2, 1, 1, 1, 1]


def test_resample_keeps_duration_and_pitch() -> None:
    t = np.arange(24000) / 24000
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = resample(tone, 24000, 48000)
    assert len(out) == 48000
    spectrum = np.abs(np.fft.rfft(out))
    assert np.argmax(spectrum) == pytest.approx(440, abs=1)  # 1 s of audio: bin == Hz


def test_limit_peak() -> None:
    quiet = np.full((4, 1), 0.5, dtype=np.float32)
    assert limit_peak(quiet)[1] == 1.0
    loud, gain = limit_peak(quiet * 4)
    assert gain < 1.0 and np.max(np.abs(loud)) == pytest.approx(10 ** (-1 / 20))


def test_fade_ramps_both_ends() -> None:
    clip = np.ones(1000, dtype=np.float32)
    out = fade(clip, rate=1000, seconds=0.01)
    assert out[0] == 0.0 and out[-1] == 0.0 and out[500] == 1.0
    assert clip[0] == 1.0  # the input is not modified
