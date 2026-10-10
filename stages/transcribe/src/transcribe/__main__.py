"""Stage 03 — transcribe: English ASR with word timestamps (WhisperX).

Whisper large-v3 runs through the ROCm build of CTranslate2 on the GPU
(int8_float16, batched), falling back to the CPU (int8) when the GPU fails;
word alignment (wav2vec2) runs on the GPU through torch.
"""

import gc
from importlib import metadata
from typing import Any

from stagekit import StageContext, StageError, main, write_json
from stagekit.probe import report

from transcribe.postprocess import SAMPLE_RATE, build_transcript
from transcribe.rocm import reexec_with_rocm_libs

VERSION = metadata.version("dublador-stage-transcribe")
CPU_COMPUTE_TYPE = "int8"
# WhisperX's defaults for its pyannote VAD.
VAD_OPTIONS = {"chunk_size": 30, "vad_onset": 0.500, "vad_offset": 0.363}
DEFAULTS: dict[str, Any] = {
    "model": "large-v3",
    "device": "cuda",
    # Same accuracy as CPU int8 on the reference video, and the least VRAM; batch 4
    # keeps the system peak near 5.6 GB of 8 (docs/MODELS.md, #19).
    "compute_type": "int8_float16",
    "batch_size": 4,
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


def _cpu_vad() -> Any:
    """WhisperX puts its VAD on the ASR device, but MIOpen cannot build the LSTM
    dropout kernel of pyannote's segmentation model on gfx1102. The VAD is small;
    keep it on the CPU so only Whisper uses the GPU."""
    import torch
    from whisperx.vads import Pyannote

    return Pyannote(torch.device("cpu"), token=None, **VAD_OPTIONS)


def _transcribe(
    whisperx: Any,
    audio: Any,
    opts: dict[str, Any],
    device: str,
    compute_type: str,
    language: str,
    ctx: StageContext,
) -> dict[str, Any]:
    ctx.progress.log(
        f"transcribing {len(audio) / SAMPLE_RATE:.0f}s with {opts['model']}"
        f" on {device} ({compute_type})"
    )
    model = whisperx.load_model(
        opts["model"],
        device,
        compute_type=compute_type,
        language=language,
        asr_options={"condition_on_previous_text": False},
        vad_model=_cpu_vad(),
        vad_options=VAD_OPTIONS,
    )
    try:
        return model.transcribe(audio, batch_size=opts["batch_size"], language=language)
    finally:
        del model
        _free_gpu()


def run(ctx: StageContext) -> None:
    import whisperx

    opts = settings(ctx.config)
    language = _language(ctx.config)
    vocals = ctx.stage_dir("separate") / "vocals.wav"
    if not vocals.is_file():
        raise StageError(f"missing input: {vocals}")

    ctx.progress.progress(0, 3)
    audio = whisperx.load_audio(str(vocals), sr=SAMPLE_RATE)
    device, compute_type = opts["device"], opts["compute_type"]
    try:
        result = _transcribe(whisperx, audio, opts, device, compute_type, language, ctx)
    except RuntimeError as e:
        if device == "cpu":
            raise
        _free_gpu()
        ctx.progress.warning(f"Whisper failed on {device} ({e}); retrying on the CPU (int8)")
        result = _transcribe(whisperx, audio, opts, "cpu", CPU_COMPUTE_TYPE, language, ctx)
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
    # The weights are the same for every device; loading on the CPU just downloads them.
    whisperx.load_model(
        opts["model"], "cpu", compute_type=CPU_COMPUTE_TYPE, language=_language(config)
    )
    whisperx.load_align_model(language_code=_language(config), device="cpu")


def probe() -> dict[str, Any]:
    doc = report(VERSION, ("whisperx", "faster-whisper", "ctranslate2", "torch", "pyannote.audio"))
    try:
        import ctranslate2

        doc["gpu"]["ctranslate2_devices"] = ctranslate2.get_cuda_device_count()
    except Exception as e:  # a broken ROCm setup must not break the probe
        doc["gpu"]["ctranslate2_devices"] = 0
        doc["gpu"]["ctranslate2_error"] = str(e)
    return doc


if __name__ == "__main__":
    reexec_with_rocm_libs("transcribe")
    main("transcribe", run=run, prefetch=prefetch, probe=probe)
