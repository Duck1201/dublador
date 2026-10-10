from stagekit.gpu import wait_for_vram


def test_wait_for_vram_returns_when_enough_is_free() -> None:
    readings = iter([1.0, 2.0, 6.5])
    assert wait_for_vram(6.0, timeout=5, poll=0, free=lambda: next(readings)) == 6.5


def test_wait_for_vram_gives_up_after_timeout() -> None:
    assert wait_for_vram(6.0, timeout=0, poll=0, free=lambda: 1.0) == 1.0


def test_wait_for_vram_without_gpu() -> None:
    assert wait_for_vram(6.0, timeout=5, poll=0, free=lambda: None) is None
