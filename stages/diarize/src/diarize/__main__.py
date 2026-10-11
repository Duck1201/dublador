"""Stage 04 — diarize: who speaks when (pyannote speaker-diarization-community-1).

Every transcript word goes to the speaker turn that covers it most, and
segments are cut where the speaker changes. The model is gated: accept its
terms on Hugging Face and log in once; runs are offline afterwards.
"""

import os
from importlib import metadata
from typing import Any

import numpy as np
import soundfile
from stagekit import StageContext, StageError, main, read_json, write_json
from stagekit.probe import report

from diarize.assign import Turn, merge_small_speakers, relabel, split_by_speaker, totals

VERSION = metadata.version("dublador-stage-diarize")
MODEL = "pyannote/speaker-diarization-community-1"
MODEL_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"
DEFAULTS: dict[str, Any] = {
    "device": "cuda",
    "num_speakers": None,
    "min_speakers": None,
    "max_speakers": None,
    # Speakers with less speech are folded into the closest bigger one.
    "min_speaker_seconds": 2.0,
}
LOGIN_HINT = (
    f"{MODEL} is not in the model cache. Accept its terms at "
    f"https://huggingface.co/{MODEL}, log in with `hf auth login` (HF_HOME pointing at "
    "<models_dir>/hf) and run `dublador setup --only diarize`."
)


def settings(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("diarize", {})
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise StageError(f"unknown [diarize] key(s): {', '.join(unknown)}")
    return {**DEFAULTS, **section}


def load_pipeline(device: str) -> Any:
    import torch
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(MODEL, revision=MODEL_REVISION)
    except Exception as e:  # missing from the offline cache, or not logged in
        raise StageError(LOGIN_HINT) from e
    if pipeline is None:
        raise StageError(LOGIN_HINT)
    if device != "cpu":
        # MIOpen cannot build the LSTM kernels of pyannote's segmentation model on
        # gfx1102 (docs/MODELS.md); torch's native RNN path works on ROCm.
        torch.backends.cudnn.enabled = False
    return pipeline.to(torch.device(device))


def diarize(audio: np.ndarray, sr: int, opts: dict[str, Any], device: str) -> list[Turn]:
    import torch

    pipeline = load_pipeline(device)
    kwargs = {k: opts[k] for k in ("num_speakers", "min_speakers", "max_speakers") if opts[k]}
    waveform = torch.from_numpy(audio).unsqueeze(0)
    output = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)
    # community-1 also gives a diarization without overlapping speech, which is
    # what a single-speaker-per-word assignment needs.
    annotation = getattr(output, "exclusive_speaker_diarization", output)
    turns = [
        Turn(str(label), float(seg.start), float(seg.end))
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return turns


def run(ctx: StageContext) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    opts = settings(ctx.config)
    transcript = read_json(ctx.stage_dir("transcribe") / "transcript.json", "transcript/v1")
    data, sr = soundfile.read(
        str(ctx.stage_dir("separate") / "vocals.wav"), dtype="float32", always_2d=True
    )
    audio = data.mean(axis=1)
    ctx.progress.progress(0, 2)
    ctx.progress.log(f"diarizing {len(audio) / sr:.0f}s on {opts['device']}")
    try:
        turns = diarize(audio, sr, opts, opts["device"])
    except (RuntimeError, MemoryError) as e:
        if opts["device"] == "cpu":
            raise
        ctx.progress.warning(f"diarization failed on {opts['device']} ({e}); retrying on the CPU")
        turns = diarize(audio, sr, opts, "cpu")
    ctx.progress.progress(1, 2)

    turns = relabel(merge_small_speakers(turns, opts["min_speaker_seconds"]))
    seconds = totals(turns)
    write_json(
        ctx.out_dir / "diarization.json",
        {
            "schema": "diarization/v1",
            "model": MODEL,
            "turns": [
                {"speaker": t.speaker, "start": round(t.start, 3), "end": round(t.end, 3)}
                for t in sorted(turns, key=lambda t: t.start)
            ],
            "speakers": [
                {"id": spk, "speech_seconds": round(seconds[spk], 3)} for spk in sorted(seconds)
            ],
        },
    )
    segments = split_by_speaker(transcript["segments"], turns)
    write_json(ctx.out_dir / "transcript.json", {**transcript, "segments": segments})
    summary = ", ".join(f"{spk} {seconds[spk]:.0f}s" for spk in sorted(seconds))
    ctx.progress.log(f"{len(seconds)} speaker(s): {summary}; {len(segments)} segments")
    ctx.progress.progress(2, 2)


def prefetch(config: dict[str, Any]) -> None:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(MODEL, revision=MODEL_REVISION)
    except Exception as e:
        raise StageError(LOGIN_HINT) from e


def probe() -> dict[str, Any]:
    return report(VERSION, ("pyannote.audio", "torch", "torchcodec"))


if __name__ == "__main__":
    main("diarize", run=run, prefetch=prefetch, probe=probe)
