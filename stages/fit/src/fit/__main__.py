"""Stage 08 — fit: make each dubbed line fit its timing window (ADR-0004, ADR-0018).

Reads the synthesized clips and the translation's timing, decides a strategy per
segment (fitting.py), time-stretches with ffmpeg's rubberband filter (pitch
kept) and writes fit/v1 with the hints the fit loop needs. With --only, only
those ids are redone and the rest of fit.json is kept.
"""

import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

import soundfile
from stagekit import StageContext, StageError, atomic_path, hash_value, main, read_json, write_json
from stagekit.probe import report

from fit.fitting import attempt, decide, room, shorten_to

VERSION = metadata.version("dublador-stage-fit")
SEGMENTS_DIR = "08_fit/segments"
# Rubber Band (R2 in this ffmpeg build) settings for speech: soft transients,
# best pitch quality.
RUBBERBAND = "rubberband=tempo={tempo:.4f}:transients=mixed:pitchq=quality"


def stretch(src: Path, dst: Path, tempo: float) -> None:
    """Speed `src` up by `tempo` (pitch kept) into `dst`; a plain copy at 1x."""
    with atomic_path(dst) as tmp:
        if abs(tempo - 1.0) < 1e-3:
            shutil.copyfile(src, tmp)
            return
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src),
               "-af", RUBBERBAND.format(tempo=tempo), "-f", "wav", str(tmp)]  # fmt: skip
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise StageError(f"ffmpeg rubberband failed on {src.name}: {result.stderr.strip()}")


def _previous(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        return {s["id"]: s for s in read_json(path, "fit/v1")["segments"]}
    except Exception:  # an unreadable cache is just a cache miss
        return {}


def run(ctx: StageContext) -> None:
    fit_cfg = ctx.config.get("fit", {})
    max_speedup = float(fit_cfg.get("max_speedup", 1.25))
    max_forced = max(float(fit_cfg.get("max_forced_speedup", 1.6)), max_speedup)
    gap = float(fit_cfg.get("gap_seconds", 0.15))

    translation = read_json(ctx.stage_dir("translate") / "translation.json", "translation/v1")
    synthesis = read_json(ctx.stage_dir("synthesize") / "synthesis.json", "synthesis/v1")
    clips = {s["id"]: s for s in synthesis["segments"]}
    lines = sorted(translation["segments"], key=lambda s: s["start"])
    missing = [s["id"] for s in lines if s["id"] not in clips]
    if missing:
        raise StageError(f"no synthesized audio for {', '.join(missing[:5])}")

    out_path = ctx.out_dir / "fit.json"
    previous = _previous(out_path)
    entries: dict[str, dict[str, Any]] = {}
    for i, line in enumerate(lines):
        seg_id = line["id"]
        clip = clips[seg_id]
        next_start = lines[i + 1]["start"] if i + 1 < len(lines) else None
        available = room(line["start"], line["end"], next_start, gap)
        key = hash_value(
            {"clip": clip["input_hash"], "start": line["start"], "room": available,
             "params": [max_speedup, max_forced, gap]}
        )  # fmt: skip
        old = previous.get(seg_id)
        reusable = (
            old is not None
            and old["input_hash"] == key
            and (ctx.job_dir / old["path"]).is_file()
            and (ctx.only is None or seg_id not in ctx.only)
        )
        if reusable or (ctx.only is not None and seg_id not in ctx.only and old is not None):
            entries[seg_id] = old  # type: ignore[assignment]
            continue

        natural = float(clip["duration"])
        decision = decide(
            natural,
            line["end"] - line["start"],
            available,
            max_speedup=max_speedup,
            max_forced=max_forced,
        )
        overflow = decision.strategy == "overflow"
        tries = attempt(old["attempt"] if old else None, overflow)
        syllables = line.get("syllables")
        path = f"{SEGMENTS_DIR}/{seg_id}.wav"
        stretch(ctx.job_dir / clip["path"], ctx.job_dir / path, decision.ratio)
        fitted = soundfile.info(str(ctx.job_dir / path)).duration
        entries[seg_id] = {
            "id": seg_id,
            "path": path,
            "start": line["start"],
            "end": round(line["start"] + fitted, 3),
            "window": round(min(available, 1e6), 3),
            "natural": round(natural, 3),
            "ratio": round(decision.ratio, 3),
            "strategy": decision.strategy,
            "attempt": tries,
            "shorten_to": (
                shorten_to(syllables, natural, available)
                if overflow and tries == 1 and syllables
                else None
            ),
            "target_duration": round(available, 3) if overflow and tries >= 2 else None,
            "input_hash": key,
        }
        if overflow:
            ctx.progress.warning(
                f"{seg_id}: {natural:.1f}s of speech for a {available:.1f}s window;"
                f" sped up {decision.ratio:.2f}x (limit {max_speedup:g}x)"
            )
        ctx.progress.progress(i + 1, len(lines))

    ordered = [entries[s["id"]] for s in lines]
    write_json(
        out_path,
        {"schema": "fit/v1", "sample_rate": synthesis["sample_rate"], "segments": ordered},
    )
    counts: dict[str, int] = {}
    for e in ordered:
        counts[e["strategy"]] = counts.get(e["strategy"], 0) + 1
    ctx.progress.log(", ".join(f"{n} {k}" for k, n in sorted(counts.items())))


def probe() -> dict[str, Any]:
    return report(VERSION, ("numpy", "soundfile"), gpu=False)


if __name__ == "__main__":
    main("fit", run=run, probe=probe)
