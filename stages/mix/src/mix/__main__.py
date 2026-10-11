"""Stage 09 — mix: dubbed speech over the separated background.

Each time-fitted clip (08_fit) starts where its original line started, with a
short fade at both ends; the sum with accompaniment.wav is written as dub.wav
at the original sample rate and channel count. Residual-voice ducking,
per-clip loudness matching and loudnorm arrive in M4.
"""

import math
from importlib import metadata
from typing import Any

import numpy as np
import soundfile
from stagekit import StageContext, StageError, atomic_path, main, read_json
from stagekit.probe import report

from mix.mixing import fade, limit_peak, place, resample

VERSION = metadata.version("dublador-stage-mix")


def run(ctx: StageContext) -> None:
    fitted = read_json(ctx.stage_dir("fit") / "fit.json", "fit/v1")
    background, rate = soundfile.read(
        str(ctx.stage_dir("separate") / "accompaniment.wav"), dtype="float32", always_2d=True
    )

    clips = []
    total = len(fitted["segments"])
    for n, seg in enumerate(fitted["segments"], start=1):
        path = ctx.job_dir / seg["path"]
        if not path.is_file():
            raise StageError(f"{seg['id']}: missing {seg['path']}")
        audio, clip_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        clips.append((seg["start"], fade(resample(audio.mean(axis=1), clip_rate, rate), rate)))
        ctx.progress.progress(n, total + 1)

    speech = place(clips, rate=rate, length=len(background), channels=background.shape[1])
    mixed = speech
    mixed[: len(background)] += background
    mixed, gain = limit_peak(mixed)
    if gain < 1.0:
        ctx.progress.warning(f"mix would clip; lowered by {-20 * math.log10(gain):.1f} dB")
    overrun = (len(mixed) - len(background)) / rate
    if overrun > 0:
        ctx.progress.warning(
            f"dubbed speech runs {overrun:.1f}s past the end of the original audio"
        )

    with atomic_path(ctx.out_dir / "dub.wav") as tmp:
        soundfile.write(str(tmp), np.ascontiguousarray(mixed), rate, subtype="PCM_16")
    ctx.progress.progress(total + 1, total + 1)


def probe() -> dict[str, Any]:
    return report(VERSION, ("numpy", "scipy", "soundfile"), gpu=False)


if __name__ == "__main__":
    main("mix", run=run, probe=probe)
