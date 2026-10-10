"""Stage 03 — transcribe: English ASR with word timestamps (WhisperX).

Whisper large-v3 runs through CTranslate2 on the CPU (int8, batched); word
alignment (wav2vec2) runs on the GPU through torch. GPU ASR is tracked in #19.
"""

import gc
from importlib import metadata
from typing import Any

from stagekit import StageContext, StageError, main, write_json
from stagekit.probe import report

from transcribe.postprocess import SAMPLE_RATE, build_transcript

VERSION = metadata.version("dublador-stage-transcribe")
DEFAULTS: dict[str, Any] = {
    "model": "large-v3",
    "device": "cpu",
    "compute_type": "int8",
    "batch_size": 8,
    "align_device": "cuda",
    "silence_dbfs": -50.0,
    # Sentence-sized segments for translation and timing (seconds).
    "max_segment_seconds": 12.0,
    "split_pause_seconds": 0.6,
}


def settings(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("transcribe", {})
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise StageError(f"unknown [transcribe] key(s): {', '.join(unknown)}")
    return {**DEFAULTS, **section}


def _language(config: dict[str, Any]) -> str:
    return config.get("languages", {}).get("source", "en").split("-")[0]


def _free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(ctx: StageContext) -> None:
    import whisperx

    opts = settings(ctx.config)
    language = _language(ctx.config)
    vocals = ctx.stage_dir("separate") / "vocals.wav"
    if not vocals.is_file():
        raise StageError(f"missing input: {vocals}")

    ctx.progress.progress(0, 3)
    audio = whisperx.load_audio(str(vocals), sr=SAMPLE_RATE)
    ctx.progress.log(
        f"transcribing {len(audio) / SAMPLE_RATE:.0f}s with {opts['model']} on {opts['device']}"
    )
    model = whisperx.load_model(
        opts["model"],
        opts["device"],
        compute_type=opts["compute_type"],
        language=language,
        asr_options={"condition_on_previous_text": False},
    )
    result = model.transcribe(audio, batch_size=opts["batch_size"], language=language)
    del model
    _free_gpu()
    ctx.progress.progress(1, 3)

    align_model, align_meta = whisperx.load_align_model(
        language_code=language, device=opts["align_device"]
    )
    aligned = whisperx.align(
        result["segments"],
        align_model,
        align_meta,
        audio,
        opts["align_device"],
        return_char_alignments=False,
    )
    del align_model
    _free_gpu()
    ctx.progress.progress(2, 3)

    doc, warnings = build_transcript(
        aligned["segments"],
        audio,
        language=language,
        silence_dbfs=opts["silence_dbfs"],
        max_duration=opts["max_segment_seconds"],
        min_pause=opts["split_pause_seconds"],
    )
    for warning in warnings:
        ctx.progress.warning(warning)
    if not doc["segments"]:
        raise StageError("no speech found in vocals.wav")
    write_json(ctx.out_dir / "transcript.json", doc)
    ctx.progress.log(f"{len(doc['segments'])} segments, {len(warnings)} dropped")
    ctx.progress.progress(3, 3)


def prefetch(config: dict[str, Any]) -> None:
    import whisperx

    opts = settings(config)
    whisperx.load_model(
        opts["model"], "cpu", compute_type=opts["compute_type"], language=_language(config)
    )
    whisperx.load_align_model(language_code=_language(config), device="cpu")


def probe() -> dict[str, Any]:
    return report(VERSION, ("whisperx", "faster-whisper", "ctranslate2", "torch", "pyannote.audio"))


if __name__ == "__main__":
    main("transcribe", run=run, prefetch=prefetch, probe=probe)
