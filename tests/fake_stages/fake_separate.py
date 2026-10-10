"""Fake `separate`: vocals and accompaniment are copies of the extracted audio."""

from fake_common import copy_wav, maybe_fail
from stagekit import StageContext, main


def run(ctx: StageContext) -> None:
    maybe_fail(ctx)
    audio = ctx.stage_dir("extract") / "audio.wav"
    for i, name in enumerate(("vocals.wav", "accompaniment.wav"), start=1):
        copy_wav(audio, ctx.out_dir / name)
        ctx.progress.progress(i, 2)


if __name__ == "__main__":
    main("separate", run=run)
