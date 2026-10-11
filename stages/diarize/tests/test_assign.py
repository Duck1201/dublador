"""Turn cleanup and word-level speaker assignment."""

from diarize.assign import Turn, merge_small_speakers, relabel, speaker_at, split_by_speaker


def words(*spans: tuple[str, float, float]) -> list[dict]:
    return [{"word": w, "start": s, "end": e, "score": 0.9} for w, s, e in spans]


def test_split_cuts_where_the_speaker_changes() -> None:
    turns = [Turn("A", 0.0, 2.0), Turn("B", 2.0, 4.0)]
    seg = {"id": "seg_0001", "start": 0.2, "end": 3.5, "text": "hi there yes",
           "speaker": None,
           "words": words(("hi", 0.2, 0.6), ("there.", 0.8, 1.5), ("Yes", 2.4, 3.5))}  # fmt: skip
    out = split_by_speaker([seg, {**seg, "id": "seg_0002", "start": 3.0, "end": 3.5,
                                  "words": words(("ok", 3.0, 3.5))}], turns)  # fmt: skip
    assert [(s["id"], s["speaker"], s["text"], s["start"], s["end"]) for s in out] == [
        ("seg_0001", "A", "hi there.", 0.2, 1.5),
        ("seg_0002", "B", "Yes", 2.4, 3.5),
        ("seg_0003", "B", "ok", 3.0, 3.5),
    ]
    assert list(out[0]) == ["id", "start", "end", "text", "speaker", "words"]


def test_word_outside_every_turn_goes_to_the_nearest() -> None:
    turns = [Turn("A", 0.0, 1.0), Turn("B", 5.0, 6.0)]
    assert speaker_at(1.5, 1.8, turns) == "A"
    assert speaker_at(4.0, 4.5, turns) == "B"
    assert speaker_at(0.0, 1.0, []) is None


def test_small_speakers_merge_into_the_closest_big_one() -> None:
    turns = [Turn("A", 0, 10), Turn("X", 10.2, 10.8), Turn("B", 20, 30)]
    merged = merge_small_speakers(turns, min_seconds=2.0)
    assert [t.speaker for t in merged] == ["A", "A", "B"]
    assert merge_small_speakers([Turn("X", 0, 1)], min_seconds=2.0)[0].speaker == "X"


def test_relabel_by_first_appearance() -> None:
    turns = [Turn("spk_7", 5, 6), Turn("spk_3", 0, 1), Turn("spk_7", 7, 8)]
    assert [t.speaker for t in relabel(turns)] == ["SPK_01", "SPK_00", "SPK_01"]
