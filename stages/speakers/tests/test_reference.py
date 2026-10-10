import numpy as np
from speakers.reference import best_reference, candidates, speech_seconds

SR = 1000


def seg(
    start: float, end: float, text: str = "words", speaker: str | None = None, score: float = 0.9
) -> dict:
    return {
        "id": f"seg_{int(start * 10):04d}",
        "start": start,
        "end": end,
        "text": text,
        "speaker": speaker,
        "words": [{"word": text, "start": start, "end": end, "score": score}],
    }


def test_speech_seconds_defaults_to_single_speaker() -> None:
    assert speech_seconds([seg(0, 2), seg(3, 4.5)]) == {"SPK_00": 3.5}


def test_candidates_join_close_segments_only() -> None:
    segs = [seg(0, 3, "a"), seg(3.2, 7, "b"), seg(9, 12, "c"), seg(12.1, 16, "d")]
    found = candidates(segs, min_seconds=6, max_seconds=15, max_gap=0.5)
    spans = {(c.start, c.end, c.text) for c in found}
    assert (0, 7, "a b") in spans
    assert (9, 16, "c d") in spans
    assert all(not (c.start == 0 and c.end > 7) for c in found)  # 2 s gap breaks the run


def test_best_reference_prefers_clean_voice() -> None:
    n = 40 * SR
    rng = np.random.default_rng(0)
    vocals = rng.normal(0, 0.1, n).astype(np.float32)
    music = np.zeros(n, dtype=np.float32)
    music[: 20 * SR] = 0.2  # loud music under the first half
    segs = [seg(0, 10), seg(20, 30)]
    ref, warning = best_reference(
        segs, vocals, music, SR, speaker="SPK_00",
        min_seconds=6, fallback_min_seconds=3, max_seconds=15, max_gap=0.5,
    )  # fmt: skip
    assert warning is None
    assert ref is not None and ref.start == 20


def test_short_speech_falls_back_then_gives_up() -> None:
    audio = np.ones(20 * SR, dtype=np.float32) * 0.1
    kwargs = {
        "speaker": "SPK_00",
        "min_seconds": 6,
        "fallback_min_seconds": 3,
        "max_seconds": 15,
        "max_gap": 0.5,
    }
    ref, warning = best_reference([seg(0, 4)], audio, audio * 0, SR, **kwargs)
    assert ref is not None and warning and "shorter" in warning
    ref, warning = best_reference([seg(0, 2)], audio, audio * 0, SR, **kwargs)
    assert ref is None and warning and "bank voice" in warning
