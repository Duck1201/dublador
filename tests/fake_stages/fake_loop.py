"""Fake translate/synthesize/fit for the fit loop: they record each call (with
--only) in <job>/calls.log, and fit follows a fixed script of overflows."""

import json

from stagekit import StageContext, atomic_write_text

# Overflows in a row before each segment fits: seg_0002 is fixed by the shorter
# translation, seg_0003 only by the target duration.
OVERFLOWS = {"seg_0001": 0, "seg_0002": 1, "seg_0003": 2}


def record(ctx: StageContext) -> None:
    only = ",".join(sorted(ctx.only)) if ctx.only else "-"
    with (ctx.job_dir / "calls.log").open("a") as log:
        log.write(f"{ctx.name}:{only}\n")


def translate(ctx: StageContext) -> None:
    record(ctx)
    n = len((ctx.job_dir / "calls.log").read_text().splitlines())
    atomic_write_text(ctx.out_dir / "translation.json", json.dumps({"version": n}))


def synthesize(ctx: StageContext) -> None:
    record(ctx)
    n = len((ctx.job_dir / "calls.log").read_text().splitlines())
    atomic_write_text(ctx.out_dir / "synthesis.json", json.dumps({"version": n}))


def fit(ctx: StageContext) -> None:
    record(ctx)
    path = ctx.out_dir / "fit.json"
    old = {s["id"]: s for s in json.loads(path.read_text())["segments"]} if path.is_file() else {}
    segments = []
    for seg_id, overflows in OVERFLOWS.items():
        if ctx.only is not None and seg_id not in ctx.only:
            segments.append(old[seg_id])
            continue
        tries = old[seg_id]["attempt"] if seg_id in old else 0
        overflow = tries < overflows
        attempt = tries + 1 if overflow else tries
        if overflow:
            ctx.progress.warning(f"{seg_id} overflows")
        segments.append({
            "id": seg_id, "strategy": "overflow" if overflow else "fits", "attempt": attempt,
            "shorten_to": 5 if overflow and attempt == 1 else None,
            "target_duration": 1.0 if overflow and attempt >= 2 else None,
        })  # fmt: skip
    atomic_write_text(path, json.dumps({"segments": segments}))
