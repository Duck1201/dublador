"""Fake `mix`: the dub is a copy of the accompaniment."""

from fake_common import copy_wav, maybe_fail
from stagekit import StageContext, main


def run(ctx: StageContext) -> None:
    maybe_fail(ctx)
    print("library noise on stdout")  # must not break the JSON-lines protocol
    ctx.progress.log("mixing")
    ctx.progress.warning("fake mix warning")
    copy_wav(ctx.stage_dir("separate") / "accompaniment.wav", ctx.out_dir / "dub.wav")
    ctx.progress.progress(1, 1)


if __name__ == "__main__":
    main("mix", run=run)
