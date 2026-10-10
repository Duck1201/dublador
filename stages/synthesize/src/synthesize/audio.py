"""Post-processing of synthesized segments, independent of the TTS backend."""

import numpy as np

FRAME_SECONDS = 0.02
THRESHOLD_DB = -40.0  # below the loudest frame
PAD_SECONDS = 0.05  # keep word onsets and release tails intact


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Drop leading and trailing silence; pauses inside the sentence stay."""
    frame = max(1, int(FRAME_SECONDS * sample_rate))
    n = len(audio) // frame
    if n == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    peak = rms.max()
    if peak <= 0:
        return audio
    loud = np.flatnonzero(20 * np.log10(np.maximum(rms, 1e-12) / peak) > THRESHOLD_DB)
    pad = int(PAD_SECONDS * sample_rate)
    start = max(0, loud[0] * frame - pad)
    end = min(len(audio), (loud[-1] + 1) * frame + pad)
    return audio[start:end]
