"""Placing dubbed clips on the timeline (pure numpy)."""

from collections.abc import Iterable
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

PEAK_LIMIT = 10 ** (-1 / 20)  # -1 dBFS


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    ratio = Fraction(dst_rate, src_rate).limit_denominator(1000)
    return resample_poly(audio, ratio.numerator, ratio.denominator, axis=0).astype(np.float32)


def place(
    clips: Iterable[tuple[float, np.ndarray]], *, rate: int, length: int, channels: int
) -> np.ndarray:
    """Sum mono clips into a (frames, channels) track at their start times (seconds).

    The track grows if a clip runs past `length`."""
    track = np.zeros((length, channels), dtype=np.float32)
    for start, clip in clips:
        begin = max(0, round(start * rate))
        end = begin + len(clip)
        if end > len(track):
            track = np.concatenate(
                [track, np.zeros((end - len(track), channels), dtype=np.float32)]
            )
        track[begin:end] += clip[:, None]
    return track


def limit_peak(track: np.ndarray, limit: float = PEAK_LIMIT) -> tuple[np.ndarray, float]:
    """Scale the whole track down if it would clip. Returns (track, gain applied)."""
    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak <= limit:
        return track, 1.0
    gain = limit / peak
    return track * gain, gain
