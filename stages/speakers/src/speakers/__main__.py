"""Stage 05 — speakers (v0): one voice reference per speaker, no models yet.

M1 assumes a single speaker (SPK_00 when the transcript has no speaker
labels). M2 adds diarization-aware speakers, gender and the voice bank.
"""

from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm
import soundfile
from stagekit import StageContext, StageError, atomic_path, main, read_json, write_json
from stagekit.probe import report

from speakers.reference import best_reference, speaker_of, speech_seconds

VERSION = metadata.version("dublador-stage-speakers")
DEFAULTS: dict[str, Any] = {
    "min_reference_seconds": 6.0,
    "fallback_min_reference_seconds": 3.0,
    "max_reference_seconds": 15.0,
    "max_gap_seconds": 0.5,
    "reference_lufs": -20.0,
}
PEAK_LIMIT = 10 ** (-1 / 20)  # -1 dBFS


def settings(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("speakers", {})
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise StageError(f"unknown [speakers] key(s): {', '.join(unknown)}")
    return {**DEFAULTS, **section}


def _mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def normalize(clip: np.ndarray, sr: int, lufs: float) -> np.ndarray:
    meter = pyloudnorm.Meter(sr)
    loudness = meter.integrated_loudness(clip) if len(clip) >= sr * 0.5 else float("-inf")
    if np.isfinite(loudness):
        clip = pyloudnorm.normalize.loudness(clip, loudness, lufs)
    peak = float(np.max(np.abs(clip))) if clip.size else 0.0
    if peak > PEAK_LIMIT:
        clip = clip * (PEAK_LIMIT / peak)
    return clip.astype(np.float32)


def run(ctx: StageContext) -> None:
    opts = settings(ctx.config)
    transcript = read_json(ctx.stage_dir("transcribe") / "transcript.json", "transcript/v1")
    segments = transcript["segments"]
    vocals, sr = _mono(ctx.stage_dir("separate") / "vocals.wav")
    accompaniment, sr_acc = _mono(ctx.stage_dir("separate") / "accompaniment.wav")
    if sr_acc != sr:
        raise StageError(f"vocals ({sr} Hz) and accompaniment ({sr_acc} Hz) sample rates differ")

    totals = speech_seconds(segments)
    speakers_out = []
    order = sorted(
        totals, key=lambda spk: min(s["start"] for s in segments if speaker_of(s) == spk)
    )
    for i, speaker in enumerate(order, start=1):
        ref, warning = best_reference(
            segments,
            vocals,
            accompaniment,
            sr,
            speaker=speaker,
            min_seconds=opts["min_reference_seconds"],
            fallback_min_seconds=opts["fallback_min_reference_seconds"],
            max_seconds=opts["max_reference_seconds"],
            max_gap=opts["max_gap_seconds"],
        )
        if warning:
            ctx.progress.warning(warning)
        reference = None
        if ref is not None:
            rel = f"05_speakers/refs/{speaker}.wav"
            clip = normalize(
                vocals[int(ref.start * sr) : int(ref.end * sr)], sr, opts["reference_lufs"]
            )
            with atomic_path(ctx.job_dir / rel) as tmp:
                soundfile.write(str(tmp), clip, sr, subtype="PCM_16")
            reference = {
                "path": rel,
                "start": round(ref.start, 3),
                "end": round(ref.end, 3),
                "text": ref.text,
            }
            ctx.progress.log(f"{speaker}: {ref.duration:.1f}s reference at {ref.start:.1f}s")
        speakers_out.append(
            {
                "id": speaker,
                "speech_seconds": round(totals[speaker], 3),
                "gender": "unknown",
                "gender_confidence": None,
                "strategy": "clone" if reference else "bank",
                "voice": None,
                "reference": reference,
            }
        )
        ctx.progress.progress(i, len(order))

    if not speakers_out:
        raise StageError("the transcript has no segments")
    write_json(ctx.out_dir / "speakers.json", {"schema": "speakers/v1", "speakers": speakers_out})


def probe() -> dict[str, Any]:
    return report(VERSION, ("numpy", "soundfile", "pyloudnorm"), gpu=False)


if __name__ == "__main__":
    main("speakers", run=run, probe=probe)
