"""GPU helpers for stages that use torch (imported lazily)."""

import time
from collections.abc import Callable


def free_vram_gb() -> float | None:
    """Free memory on GPU 0 as seen by torch, or None without a GPU."""
    import torch  # pyright: ignore[reportMissingImports]  # only in stage venvs

    if not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


def wait_for_vram(
    min_free_gb: float,
    *,
    timeout: float = 60.0,
    poll: float = 1.0,
    free: Callable[[], float | None] = free_vram_gb,
) -> float | None:
    """Wait until `min_free_gb` are free (the previous stage may still be
    releasing memory). Returns the last reading; never raises on timeout."""
    deadline = time.monotonic() + timeout
    while True:
        current = free()
        if current is None or current >= min_free_gb or time.monotonic() >= deadline:
            return current
        time.sleep(poll)
