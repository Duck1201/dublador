import numpy as np
from speakers.reference import TAIL, best_reference, candidates, speech_seconds

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
    spans = {(c.start, round(c.end, 3), c.text) for c in found}
    assert (0, 7 + TAIL, "a b") in spans
    assert (9, 16 + TAIL, "c d") in spans
    assert all(not (c.start == 0 and c.end > 7 + TAIL) for c in found)  # 2 s gap breaks the run


def test_candidates_cut_at_word_boundaries() -> None:
    words = [
        {"word": w, "start": t, "end": t + 0.85, "score": 0.9}
        for w, t in (("one", 0.0), ("two", 0.9), ("three", 1.8), ("four", 2.7), ("five", 3.6))
    ]
    segs = [{"id": "seg_0001", "start": 0.0, "end": 4.45, "text": "x", "speaker": None,
             "words": words}]  # fmt: skip
    found = candidates(segs, min_seconds=3, max_seconds=3.6, max_gap=0.5)
    assert {(c.start, round(c.end, 3), c.text) for c in found} == {
        (0.0, 3.6, "one two three four"),  # the tail stops where the next word starts
        (0.9, round(4.45 + TAIL, 3), "two three four five"),
    }


def test_best_reference_prefers_clean_voice() -> None:
    n = 40 * SR
    rng = np.random.default_rng(0)
    vocals = rng.normal(0, 0.1, n).astype(np.float32)
    music = np.zeros(n, dtype=np.float32)
    music[: 20 * SR] = 0.2  # loud music under the first half
    segs = [seg(0, 10), seg(20, 30)]
    ref, warning = best_reference(
        segs, vocals, music, SR, speaker="SPK_00",
        min_seconds=6, fallback_min_seconds=3, max_seconds=15, target_seconds=10,
        max_gap=0.5,
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
        "target_seconds": 10,
        "max_gap": 0.5,
    }
    ref, warning = best_reference([seg(0, 4)], audio, audio * 0, SR, **kwargs)
    assert ref is not None and warning and "shorter" in warning
    ref, warning = best_reference([seg(0, 2)], audio, audio * 0, SR, **kwargs)
    assert ref is None and warning and "bank voice" in warning


def test_reference_prefers_clean_edges() -> None:
    # "Unknown or when they" stops mid-phrase; "It is simple." ends a sentence.
    spans = [
        ("It", 0.0, 0.4), ("is", 0.5, 0.8), ("simple.", 0.9, 1.6), ("Unknown", 2.0, 2.6),
        ("or", 2.7, 2.9), ("when", 3.0, 3.3), ("they", 3.35, 3.6), ("do", 3.65, 4.0),
    ]  # fmt: skip
    words = [{"word": w, "start": a, "end": b, "score": 0.9} for w, a, b in spans]
    segs = [{"id": "seg_0001", "start": 0.0, "end": 4.0, "text": "x", "speaker": None,
             "words": words}]  # fmt: skip
    found = candidates(segs, min_seconds=1.5, max_seconds=4.0, max_gap=0.5)
    clean = {c.text for c in found if c.clean_edges}
    assert "It is simple." in clean
    assert "Unknown or when they" not in clean  # ends right before "do"
    audio = np.ones(5 * SR, dtype=np.float32) * 0.1
    ref, _ = best_reference(
        segs, audio, audio * 0, SR, speaker="SPK_00", min_seconds=1.5, fallback_min_seconds=1.0,
        max_seconds=4.0, target_seconds=3.5, max_gap=0.5,
    )  # fmt: skip
    assert ref is not None and ref.clean_edges
