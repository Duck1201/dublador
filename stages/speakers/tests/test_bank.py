"""Bank-voice choice: by gender, stable, no repeats until a gender runs out."""

from speakers.bank import assign


def test_distinct_voices_per_gender() -> None:
    chosen = assign([("SPK_00", "male"), ("SPK_01", "female"), ("SPK_02", "male")], per_gender=4)
    assert chosen == {"SPK_00": "male_01", "SPK_01": "female_01", "SPK_02": "male_02"}


def test_unknown_alternates_and_voices_wrap() -> None:
    chosen = assign([("SPK_00", "unknown"), ("SPK_01", "unknown")], per_gender=4)
    assert chosen == {"SPK_00": "male_01", "SPK_01": "female_01"}
    wrapped = assign([(f"SPK_0{i}", "female") for i in range(3)], per_gender=2)
    assert list(wrapped.values()) == ["female_01", "female_02", "female_01"]
