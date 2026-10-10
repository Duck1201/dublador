import numpy as np
import pytest
from stagekit import validate
from transcribe.postprocess import SAMPLE_RATE, build_transcript, is_repetitive, rms_dbfs


def tone(seconds: float, amplitude: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def seg(start: float, end: float, text: str, words: list | None = None) -> dict:
    return {"start": start, "end": end, "text": text, "words": words or []}


def test_rms_dbfs() -> None:
    audio = np.concatenate([np.zeros(SAMPLE_RATE, dtype=np.float32), tone(1)])
    assert rms_dbfs(audio, 0, 1) == float("-inf")
    assert rms_dbfs(audio, 1, 2) == pytest.approx(-23.0, abs=0.1)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Thank you. Thank you. Thank you. Thank you.", True),
        ("no no no no no", True),
        ("I said no, no, no to that.", False),
        ("Welcome back to the channel.", False),
    ],
)
def test_is_repetitive(text: str, expected: bool) -> None:
    assert is_repetitive(text) is expected


def test_build_transcript_filters_and_numbers() -> None:
    audio = np.concatenate([tone(4), np.zeros(2 * SAMPLE_RATE, dtype=np.float32), tone(6)])
    raw = [
        seg(
            0.0,
            1.5,
            " Hello there. ",
            [{"word": "Hello", "start": 0.1, "end": 0.5, "score": 0.91}, {"word": "there."}],
        ),
        seg(4.2, 5.8, " Thanks for watching!"),  # over silence
        seg(6.0, 7.0, "Okay."),
        seg(7.0, 8.0, "Okay."),
        seg(8.0, 9.0, "Okay."),  # third identical in a row
        seg(9.0, 10.0, "la la la la la"),
        seg(10.0, 11.0, " ... "),
        {"start": None, "end": None, "text": "unaligned", "words": []},
    ]
    doc, warnings = build_transcript(raw, audio, language="en", silence_dbfs=-50)
    validate(doc, "transcript/v1")
    assert [s["id"] for s in doc["segments"]] == ["seg_0001", "seg_0002", "seg_0003"]
    assert [s["text"] for s in doc["segments"]] == ["Hello there.", "Okay.", "Okay."]
    assert doc["segments"][0]["words"] == [
        {"word": "Hello", "start": 0.1, "end": 0.5, "score": 0.91},
        {"word": "there."},
    ]
    reasons = " | ".join(warnings)
    for reason in ("text over silence", "same text three times", "repetition loop", "no words"):
        assert reason in reasons


def words(*spec: tuple[str, float, float]) -> list[dict]:
    return [{"word": w, "start": s, "end": e, "score": 0.9} for w, s, e in spec]


def test_split_at_sentence_end_and_pause() -> None:
    from transcribe.postprocess import split_segment

    raw = seg(
        0.0,
        9.0,
        "Hi there. how are you doing today",
        words(
            ("Hi", 0.0, 0.3), ("there.", 0.35, 0.8),
            ("how", 1.0, 1.2), ("are", 1.25, 1.4), ("you", 1.45, 1.6),
            ("doing", 3.0, 3.4), ("today", 3.45, 3.9),
        ),
    )  # fmt: skip
    pieces = split_segment(raw, max_duration=12, min_pause=0.6)
    assert [p["text"] for p in pieces] == ["Hi there.", "how are you", "doing today"]
    assert (pieces[1]["start"], pieces[1]["end"]) == (1.0, 1.6)


def test_long_piece_cut_at_widest_gap() -> None:
    from transcribe.postprocess import split_segment

    spec = [(f"w{i}", i * 1.0, i * 1.0 + 0.8) for i in range(10)]
    spec[6] = ("w6", 6.3, 7.1)  # widest gap (0.5 s), still below min_pause
    pieces = split_segment(seg(0, 10, "", words(*spec)), max_duration=6, min_pause=0.6)
    assert [p["text"] for p in pieces] == ["w0 w1 w2 w3 w4 w5", "w6 w7 w8 w9"]


def test_segment_without_word_times_is_kept() -> None:
    from transcribe.postprocess import split_segment

    raw = seg(1.0, 2.0, "Hm.", [{"word": "Hm."}])
    assert split_segment(raw, max_duration=12, min_pause=0.6) == [raw]
