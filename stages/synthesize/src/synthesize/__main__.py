"""Stage 07 — synthesize: one pt-BR WAV per segment, cloning each speaker's voice.

The TTS model is a pluggable backend ([synthesize] backend). Results are
cached per segment by an input hash (text, voice reference, backend
options); synthesis.json is checkpointed so long jobs resume mid-way.
"""

from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import soundfile
from stagekit import StageContext, StageError, atomic_path, main, read_json, write_json
from stagekit.gpu import wait_for_vram
from stagekit.probe import report

from synthesize import backends, voicebank
from synthesize.audio import trim_silence
from synthesize.plan import MissingBankVoice, jobs, max_seconds, ran_away, split, voices

VERSION = metadata.version("dublador-stage-synthesize")
DEFAULTS: dict[str, Any] = {
    "backend": "omnivoice",
    "min_free_vram_gb": 5.0,  # wait for the previous stage to release the GPU
    "checkpoint_every": 10,
}
SEGMENTS_DIR = "07_synthesize/segments"


def settings(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(stage options, backend options): unknown keys go to the backend, which validates them."""
    section = dict(config.get("synthesize", {}))
    stage = {k: section.pop(k, v) for k, v in DEFAULTS.items()}
    return stage, section


def _previous(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        doc = read_json(path, "synthesis/v1")
    except Exception:  # an unreadable cache is just a cache miss
        return {}
    return {s["id"]: {**s, "_sample_rate": doc["sample_rate"]} for s in doc["segments"]}


def _entry(
    job_id: str, duration: float, input_hash: str, target: float | None = None
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": job_id,
        "path": f"{SEGMENTS_DIR}/{job_id}.wav",
        "duration": round(duration, 3),
        "input_hash": input_hash,
    }
    if target is not None:
        entry["target_duration"] = target
    return entry


def _durations(
    ctx: StageContext,
    translation: dict[str, Any],
    by_speaker: dict[str, Any],
    backend_key: dict[str, Any],
    previous: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Target lengths for the backend: what the fit loop asks for the --only ids,
    plus earlier targets while the rest of the segment's inputs are unchanged."""
    earlier = {i: p["target_duration"] for i, p in previous.items() if p.get("target_duration")}
    sticky = {
        j.id: j.forced_duration
        for j in jobs(translation, by_speaker, backend_key, earlier)
        if j.forced_duration is not None and previous[j.id]["input_hash"] == j.input_hash
    }
    requested: dict[str, float] = {}
    fit_path = ctx.stage_dir("fit") / "fit.json"
    if ctx.only is not None and fit_path.is_file():
        for seg in read_json(fit_path, "fit/v1")["segments"]:
            if seg["id"] in ctx.only and seg.get("target_duration"):
                requested[seg["id"]] = seg["target_duration"]
    return {**sticky, **requested}


def run(ctx: StageContext) -> None:
    opts, backend_opts = settings(ctx.config)
    target = ctx.config.get("languages", {}).get("target", "pt-BR")
    translation = read_json(ctx.stage_dir("translate") / "translation.json", "translation/v1")
    speakers = read_json(ctx.stage_dir("speakers") / "speakers.json", "speakers/v1")
    try:
        backend = backends.create(
            opts["backend"],
            backend_opts,
            language=target,
            models_dir=ctx.config["paths"]["models_dir"],
        )
    except ValueError as e:
        raise StageError(str(e)) from e

    models_dir = Path(ctx.config["paths"]["models_dir"])
    try:
        by_speaker = voices(speakers, ctx.job_dir, models_dir)
    except MissingBankVoice as e:
        raise StageError(
            f"bank voice {e} is missing from {voicebank.bank_dir(models_dir)};"
            " run `dublador setup --only synthesize`"
        ) from e
    missing = sorted({s["speaker"] for s in translation["segments"]} - set(by_speaker))
    if missing:
        raise StageError(f"no voice reference or bank voice for {', '.join(missing)}")

    backend_key = {"backend": backend.name, **backend.settings()}
    out_path = ctx.out_dir / "synthesis.json"
    previous = _previous(out_path)
    durations = (
        _durations(ctx, translation, by_speaker, backend_key, previous)
        if backend.supports_target_duration
        else {}
    )
    all_jobs = jobs(translation, by_speaker, backend_key, durations)
    existing = {p.stem for p in (ctx.job_dir / SEGMENTS_DIR).glob("*.wav")}
    todo, keep = split(all_jobs, previous, existing, ctx.only)
    ctx.progress.log(f"{len(todo)} to synthesize, {len(keep)} cached")

    entries = {
        j.id: _entry(j.id, previous[j.id]["duration"], j.input_hash, j.forced_duration)
        for j in keep
    }
    sample_rate = next((previous[j.id]["_sample_rate"] for j in keep), 0)

    def checkpoint() -> None:
        ordered = [entries[j.id] for j in all_jobs if j.id in entries]
        write_json(
            out_path,
            {
                "schema": "synthesis/v1",
                "backend": backend.name,
                "sample_rate": sample_rate or 1,
                "segments": ordered,
            },
        )

    if todo:
        free = wait_for_vram(opts["min_free_vram_gb"])
        if free is not None and free < opts["min_free_vram_gb"]:
            ctx.progress.warning(f"only {free:.1f} GB of VRAM free; loading {backend.name} anyway")
        ctx.progress.log(f"loading {backend.name}")
        backend.load()
        prepared: dict[str, Any] = {}
        for n, job in enumerate(todo, start=1):
            voice = by_speaker[job.speaker]
            if job.speaker not in prepared:
                prepared[job.speaker] = backend.prepare_voice(str(voice.ref_wav), voice.ref_text)
            budget = max(max_seconds(job), (job.forced_duration or 0) + 1.0)
            for attempt in (1, 2):  # sampling is random: a runaway rarely repeats
                audio = backend.synthesize(
                    job.text,
                    prepared[job.speaker],
                    target_duration=job.forced_duration,
                    max_seconds=budget,
                )
                if audio.size == 0 or not np.isfinite(audio).all():
                    raise StageError(f"{job.id}: {backend.name} produced no valid audio")
                if not ran_away(len(audio) / backend.sample_rate, budget):
                    break
                if attempt == 2:
                    ctx.progress.warning(
                        f"{job.id}: {backend.name} did not finish the sentence within"
                        f" {budget:.1f} s twice; keeping the cut-off audio"
                    )
            if sample_rate and backend.sample_rate != sample_rate:
                raise StageError(
                    f"{backend.name} sample rate changed ({sample_rate} -> {backend.sample_rate})"
                )
            sample_rate = backend.sample_rate
            audio = trim_silence(audio, sample_rate)
            with atomic_path(ctx.job_dir / SEGMENTS_DIR / f"{job.id}.wav") as tmp:
                soundfile.write(str(tmp), audio, sample_rate, subtype="PCM_16")
            entries[job.id] = _entry(
                job.id, len(audio) / sample_rate, job.input_hash, job.forced_duration
            )
            ctx.progress.progress(n, len(todo))
            if n % opts["checkpoint_every"] == 0:
                checkpoint()
    checkpoint()
    if not todo:
        ctx.progress.progress(1, 1)


def prefetch(config: dict[str, Any]) -> None:
    opts, backend_opts = settings(config)
    cls = backends.backend_class(opts["backend"])
    cls.prefetch(backend_opts)
    models_dir = Path(config["paths"]["models_dir"])
    if not voicebank.missing(models_dir) or not hasattr(cls, "design_voice"):
        return
    target = config.get("languages", {}).get("target", "pt-BR")
    backend = backends.create(
        opts["backend"], backend_opts, language=target, models_dir=str(models_dir)
    )
    backend.load()
    voicebank.build(models_dir, backend.design_voice)  # pyright: ignore[reportAttributeAccessIssue]


def probe() -> dict[str, Any]:
    return report(VERSION, ("omnivoice", "transformers", "torch"))


if __name__ == "__main__":
    main("synthesize", run=run, prefetch=prefetch, probe=probe)
