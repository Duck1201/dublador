"""Helpers for fake stages: same contract as real stages, no models."""

import os
import shutil
from pathlib import Path

from stagekit import StageContext, StageError, atomic_path


def maybe_fail(ctx: StageContext) -> None:
    if os.environ.get("FAKE_FAIL") == ctx.name:
        raise StageError(f"forced failure in {ctx.name}")


def copy_wav(src: Path, dst: Path) -> None:
    with atomic_path(dst) as tmp:
        shutil.copyfile(src, tmp)
