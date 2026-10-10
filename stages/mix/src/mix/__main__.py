"""Stage 09 — mix (v0): dubbed speech over the separated background.

Each synthesized clip starts where its original line started; the sum with
accompaniment.wav is written as dub.wav at the original sample rate and
channel count. Residual-voice ducking, per-clip loudness matching and
loudnorm arrive in M4.
"""

import math
from importlib import metadata
from typing import Any

import numpy as np
import soundfile
from stagekit import StageContext, StageError, atomic_path, main, read_json
from stagekit.probe import report

from mix.mixing import limit_peak, place, resample

VERSION = metadata.version("dublador-stage-mix")


def run(ctx: StageContext) -> None:
    synthesis = read_json(ctx.stage_dir("synthesize") / "synthesis.json", "synthesis/v1")
    translation = read_json(ctx.stage_dir("translate") / "translation.json", "translation/v1")
    starts = {s["id"]: s["start"] for s in translation["segments"]}
    background, rate = soundfile.read(
        str(ctx.stage_dir("separate") / "accompaniment.wav"), dtype="float32", always_2d=True
    )

    clips = []
    total = len(synthesis["segments"])
    for n, seg in enumerate(synthesis["segments"], start=1):
        if seg["id"] not in starts:
            raise StageError(f"{seg['id']} is in synthesis.json but not in translation.json")
        audio, clip_rate = soundfile.read(
            str(ctx.job_dir / seg["path"]), dtype="float32", always_2d=True
        )
        clips.append((starts[seg["id"]], resample(audio.mean(axis=1), clip_rate, rate)))
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
