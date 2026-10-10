"""Atomic writes: a stage never leaves a half-written output behind."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def atomic_path(target: Path) -> Iterator[Path]:
    """Yield a temporary path next to `target`; move it into place on success.

    The temporary file keeps the target's suffix, so libraries that pick the
    format from the extension (soundfile, ffmpeg) work. On error it is removed.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.stem}.tmp-{os.getpid()}{target.suffix}")
    try:
        yield tmp
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_bytes(target: Path, data: bytes) -> None:
    with atomic_path(target) as tmp, tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def atomic_write_text(target: Path, text: str) -> None:
    atomic_write_bytes(target, text.encode())


def atomic_write_json(target: Path, doc: Any) -> None:
    atomic_write_text(target, json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
