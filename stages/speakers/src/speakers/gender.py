"""Speaker gender: ECAPA classifier on several clean clips, duration-weighted vote,
median F0 as a tie-breaker (docs/MODELS.md §4). Never decided per segment."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

MODEL = "JaesungHuh/voice-gender-classifier"
MODEL_REVISION = "db1222153bd60337e900be22add7af180452adc0"
MODEL_RATE = 16000
# Median F0 clearly below/above these is male/female in adult speech.
MALE_F0_MAX = 145.0
FEMALE_F0_MIN = 185.0


@dataclass(frozen=True)
class Verdict:
    gender: str  # male | female | unknown
    confidence: float | None


def vote(
    clips: Sequence[tuple[float, float]], f0: float | None, *, min_confidence: float
) -> Verdict:
    """clips: (seconds, P(female)) per clip. With p the duration-weighted mean
    P(female), the confidence is |2p - 1|; below `min_confidence` a clear median
    F0 decides instead."""
    total = sum(seconds for seconds, _ in clips)
    if total > 0:
        p_female = sum(seconds * p for seconds, p in clips) / total
        confidence = abs(2 * p_female - 1)
        if confidence >= min_confidence:
            return Verdict("female" if p_female > 0.5 else "male", round(confidence, 3))
    if f0 is not None and f0 <= MALE_F0_MAX:
        return Verdict("male", None)
    if f0 is not None and f0 >= FEMALE_F0_MIN:
        return Verdict("female", None)
    return Verdict("unknown", None)


def median_f0(
    audio: np.ndarray, sr: int, *, fmin: float = 60.0, fmax: float = 400.0
) -> float | None:
    """Median pitch of the voiced frames (autocorrelation), or None if unvoiced."""
    frame, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    if len(audio) < frame or hi >= frame:
        return None
    peak = float(np.max(np.abs(audio))) or 1.0
    pitches = []
    window = np.hanning(frame)
    for start in range(0, len(audio) - frame, hop):
        x = audio[start : start + frame] * window
        if np.sqrt(np.mean(x**2)) < 0.05 * peak:
            continue  # too quiet to be voiced
        ac = np.correlate(x, x, mode="full")[frame - 1 :]
        if ac[0] <= 0:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.4:  # periodic enough
            pitches.append(sr / lag)
    return float(np.median(pitches)) if pitches else None


def resample(audio: np.ndarray, sr: int, rate: int = MODEL_RATE) -> np.ndarray:
    if sr == rate:
        return audio.astype(np.float32)
    from scipy.signal import resample_poly

    g = np.gcd(sr, rate)
    return resample_poly(audio, rate // g, sr // g).astype(np.float32)


class Classifier:
    """ECAPA gender network on the CPU; returns P(female) per 16 kHz clip."""

    def __init__(self) -> None:
        import torch

        from speakers.ecapa_gender import ECAPA_gender

        self._torch = torch
        self._model: Any = ECAPA_gender.from_pretrained(MODEL, revision=MODEL_REVISION)
        self._model.eval()

    def p_female(self, clip: np.ndarray) -> float:
        torch = self._torch
        with torch.no_grad():
            logits = self._model(torch.from_numpy(clip).unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]
        return float(probs[1])  # the model's labels: 0 male, 1 female

    @staticmethod
    def prefetch() -> None:
        from huggingface_hub import snapshot_download

        snapshot_download(MODEL, revision=MODEL_REVISION)
