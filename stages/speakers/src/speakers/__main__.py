"""Stage 05 — speakers: per diarized speaker, a short cloning reference, the
gender (ECAPA vote over clean clips) and, without a reference, a bank voice."""

from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm
import soundfile
from stagekit import StageContext, StageError, atomic_path, main, read_json, write_json
from stagekit.probe import report

from speakers import bank
from speakers.gender import Classifier, median_f0, resample, vote
from speakers.reference import best_reference, clean_clips, speaker_of, speech_seconds

VERSION = metadata.version("dublador-stage-speakers")
# Short references (ADR-0017): a longer English clip carries its accent into pt-BR.
DEFAULTS: dict[str, Any] = {
    "min_reference_seconds": 3.0,
    "fallback_min_reference_seconds": 2.0,
    "max_reference_seconds": 4.5,
    "target_reference_seconds": 3.5,
    "max_gap_seconds": 0.5,
    "reference_lufs": -20.0,
    # Gender: several clean clips per speaker, duration-weighted vote (docs/MODELS.md §4).
    "gender_clips": 5,
    "gender_clip_min_seconds": 2.0,
    "gender_clip_max_seconds": 6.0,
    "min_gender_confidence": 0.3,
    # "bank": every speaker gets the native pt-BR bank voice of its gender closest
    # in pitch (ADR-0020); "clone": clone each speaker's own English reference,
    # with the bank only when there is no reference (ADR-0017).
    "voice_strategy": "bank",
}
STRATEGIES = ("bank", "clone")
PEAK_LIMIT = 10 ** (-1 / 20)  # -1 dBFS


def settings(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("speakers", {})
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise StageError(f"unknown [speakers] key(s): {', '.join(unknown)}")
    opts = {**DEFAULTS, **section}
    if opts["voice_strategy"] not in STRATEGIES:
        raise StageError(f"speakers.voice_strategy must be one of: {', '.join(STRATEGIES)}")
    return opts


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


def make_classifier() -> Any:
    return Classifier()


def run(ctx: StageContext) -> None:
    opts = settings(ctx.config)
    transcript = read_json(ctx.stage_dir("diarize") / "transcript.json", "transcript/v1")
    segments = transcript["segments"]
    if not segments:
        raise StageError("the transcript has no segments")
    vocals, sr = _mono(ctx.stage_dir("separate") / "vocals.wav")
    accompaniment, sr_acc = _mono(ctx.stage_dir("separate") / "accompaniment.wav")
    if sr_acc != sr:
        raise StageError(f"vocals ({sr} Hz) and accompaniment ({sr_acc} Hz) sample rates differ")

    totals = speech_seconds(segments)
    order = sorted(
        totals, key=lambda spk: min(s["start"] for s in segments if speaker_of(s) == spk)
    )
    ctx.progress.log("loading the gender classifier")
    classifier = make_classifier()
    speakers_out = []
    pitches: dict[str, float | None] = {}
    for i, speaker in enumerate(order, start=1):
        reference = _reference(ctx, opts, segments, vocals, accompaniment, sr, speaker)
        verdict, pitches[speaker] = _gender(
            opts, classifier, segments, vocals, accompaniment, sr, speaker
        )
        ctx.progress.log(
            f"{speaker}: {totals[speaker]:.0f}s of speech, {verdict.gender}"
            + (f" ({verdict.confidence:.2f})" if verdict.confidence is not None else "")
            + (f", {pitches[speaker]:.0f} Hz" if pitches[speaker] else "")
        )
        speakers_out.append(
            {
                "id": speaker,
                "speech_seconds": round(totals[speaker], 3),
                "gender": verdict.gender,
                "gender_confidence": verdict.confidence,
                "strategy": "clone" if reference and opts["voice_strategy"] == "clone" else "bank",
                "voice": None,
                "reference": reference,
            }
        )
        ctx.progress.progress(i, len(order))

    needs_bank = sorted(
        (s for s in speakers_out if s["strategy"] == "bank"), key=lambda s: -s["speech_seconds"]
    )
    if needs_bank:
        models_dir = Path(ctx.config["paths"]["models_dir"])
        available = bank.scan(models_dir, _wav_pitch)
        if not available:
            raise StageError(
                f"no voice bank in {models_dir / 'voices' / 'pt-BR'};"
                " run `dublador setup --only synthesize`"
            )
        voices = bank.assign(
            [(s["id"], s["gender"], pitches[s["id"]]) for s in needs_bank], available
        )
        pitch_of = {v.id: v.f0 for v in available}
        for spk in needs_bank:
            spk["voice"] = voices[spk["id"]]
            f0 = pitch_of[spk["voice"]]
            ctx.progress.log(
                f"{spk['id']}: bank voice {spk['voice']}" + (f" ({f0:.0f} Hz)" if f0 else "")
            )
    write_json(ctx.out_dir / "speakers.json", {"schema": "speakers/v1", "speakers": speakers_out})


def _reference(
    ctx: StageContext,
    opts: dict[str, Any],
    segments: list[dict[str, Any]],
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    sr: int,
    speaker: str,
) -> dict[str, Any] | None:
    ref, warning = best_reference(
        segments,
        vocals,
        accompaniment,
        sr,
        speaker=speaker,
        min_seconds=opts["min_reference_seconds"],
        fallback_min_seconds=opts["fallback_min_reference_seconds"],
        max_seconds=opts["max_reference_seconds"],
        target_seconds=opts["target_reference_seconds"],
        max_gap=opts["max_gap_seconds"],
    )
    if warning:
        ctx.progress.warning(warning)
    if ref is None:
        return None
    rel = f"05_speakers/refs/{speaker}.wav"
    clip = normalize(vocals[int(ref.start * sr) : int(ref.end * sr)], sr, opts["reference_lufs"])
    with atomic_path(ctx.job_dir / rel) as tmp:
        soundfile.write(str(tmp), clip, sr, subtype="PCM_16")
    ctx.progress.log(f"{speaker}: {ref.duration:.1f}s reference at {ref.start:.1f}s")
    return {"path": rel, "start": round(ref.start, 3), "end": round(ref.end, 3), "text": ref.text}


def _wav_pitch(path: Path) -> float | None:
    audio, sr = _mono(path)
    return median_f0(audio, sr)


def _gender(
    opts: dict[str, Any],
    classifier: Any,
    segments: list[dict[str, Any]],
    vocals: np.ndarray,
    accompaniment: np.ndarray,
    sr: int,
    speaker: str,
) -> tuple[Any, float | None]:
    clips = clean_clips(
        segments,
        vocals,
        accompaniment,
        sr,
        speaker=speaker,
        count=opts["gender_clips"],
        min_seconds=opts["gender_clip_min_seconds"],
        max_seconds=opts["gender_clip_max_seconds"],
        max_gap=opts["max_gap_seconds"],
    )
    audio = [vocals[int(c.start * sr) : int(c.end * sr)] for c in clips]
    scored = [
        (c.duration, classifier.p_female(resample(a, sr)))
        for c, a in zip(clips, audio, strict=True)
    ]
    f0 = median_f0(np.concatenate(audio), sr) if audio else None
    return vote(scored, f0, min_confidence=opts["min_gender_confidence"]), f0


def prefetch(config: dict[str, Any]) -> None:
    Classifier.prefetch()


def probe() -> dict[str, Any]:
    return report(VERSION, ("numpy", "soundfile", "pyloudnorm", "torch"), gpu=False)


if __name__ == "__main__":
    main("speakers", run=run, prefetch=prefetch, probe=probe)
