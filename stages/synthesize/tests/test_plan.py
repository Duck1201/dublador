"""The per-segment generation budget."""

from synthesize.plan import Job, max_seconds, ran_away


def job(text: str, window: float) -> Job:
    return Job("seg_0001", "SPK_00", text, window, "sha256:x")


def test_budget_has_a_floor_and_grows_with_window_and_text() -> None:
    assert max_seconds(job("Oi, pessoal.", 0.5)) == 4.0
    assert max_seconds(job("x" * 10, 2.0)) == 6.0
    assert max_seconds(job("x" * 100, 2.0)) == 20.0


def test_ran_away() -> None:
    assert ran_away(3.96, 4.0)
    assert not ran_away(2.5, 4.0)
