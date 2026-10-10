"""Stage 02 — separate: dialogue (vocals) vs. music and effects (accompaniment).

Uses audio-separator with a RoFormer model (docs/MODELS.md §1). vocals.wav
feeds transcription, diarization and voice references; accompaniment.wav
goes back into the final mix.
"""

import logging
import os
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

import soundfile
from stagekit import StageContext, StageError, main
from stagekit.probe import report

DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
# audio-separator's defaults for RoFormer/MDXC models; `overlap` None keeps the model's own.
_MDXC_DEFAULTS = {
    "segment_size": 256,
    "override_model_segment_size": False,
    "batch_size": 1,
    "overlap": None,
    "pitch_shift": 0,
}
OUTPUTS = {"vocals": "vocals.wav", "accompaniment": "accompaniment.wav"}
# Stem names models use for the non-vocal part, mapped to our "accompaniment".
_ACCOMPANIMENT_STEMS = ("instrumental", "accompaniment", "no vocals", "other")
VERSION = metadata.version("dublador-stage-separate")


def model_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["models_dir"]) / "audio-separator"


def _separator(config: dict[str, Any], **kwargs: Any):
    from audio_separator.separator import Separator

    return Separator(
        model_file_dir=str(model_dir(config)),
        log_level=logging.WARNING,
        **kwargs,
    )


def stem_role(path: str) -> str | None:
    """Map an audio-separator output file to "vocals" / "accompaniment"."""
    name = Path(path).stem.lower()
    if "vocals" in name and "no vocals" not in name and "no_vocals" not in name:
        return "vocals"
    if any(stem in name.replace("_", " ") for stem in _ACCOMPANIMENT_STEMS):
        return "accompaniment"
    return None


def run(ctx: StageContext) -> None:
    audio = ctx.stage_dir("extract") / "audio.wav"
    if not audio.is_file():
        raise StageError(f"missing input: {audio}")
    model = ctx.section.get("model", DEFAULT_MODEL)
    chunk = ctx.section.get("chunk_duration")
    # fp16 autocast: 2.7x faster on the RX 7600 with no measurable change in the
    # stems (73 dB vs fp32 on speech over music, 2026-10-10).
    autocast = ctx.section.get("use_autocast", True)
    mdxc = {**_MDXC_DEFAULTS, "overlap": ctx.section.get("overlap")}
    sample_rate = soundfile.info(str(audio)).samplerate
    ctx.out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=ctx.out_dir, prefix=".work-") as work:
        separator = _separator(
            ctx.config,
            output_dir=work,
            output_format="WAV",
            sample_rate=sample_rate,
            chunk_duration=chunk,
            use_soundfile=True,
            use_autocast=autocast,
            mdxc_params=mdxc,
        )
        ctx.progress.log(f"loading {model}")
        separator.load_model(model)
        ctx.progress.progress(0, 1)
        produced = separator.separate(str(audio))

        found: dict[str, Path] = {}
        for path in produced:
            role = stem_role(path)
            if role and role not in found:
                full = Path(path) if os.path.isabs(path) else Path(work) / path
                found[role] = full
        missing = sorted(set(OUTPUTS) - set(found))
        if missing:
            names = ", ".join(Path(p).name for p in produced)
            raise StageError(f"model {model} produced no {', '.join(missing)} stem ({names})")
        # Same filesystem: each output appears complete or not at all.
        for role, name in OUTPUTS.items():
            os.replace(found[role], ctx.out_dir / name)
    ctx.progress.progress(1, 1)


def prefetch(config: dict[str, Any]) -> None:
    model = config.get("separate", {}).get("model", DEFAULT_MODEL)
    model_dir(config).mkdir(parents=True, exist_ok=True)
    _separator(config, info_only=True).download_model_files(model)


def probe() -> dict[str, Any]:
    return report(VERSION, ("audio-separator", "torch", "onnxruntime"))


if __name__ == "__main__":
    main("separate", run=run, prefetch=prefetch, probe=probe)
