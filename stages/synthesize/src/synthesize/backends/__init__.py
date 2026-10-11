"""Pluggable TTS backends. The bake-off (docs/MODELS.md §6.2) swaps only this."""

import importlib
from typing import Any, Protocol

import numpy as np


class TTSBackend(Protocol):
    """What the synthesize stage needs from a TTS model."""

    name: str
    sample_rate: int
    supports_target_duration: bool

    def load(self) -> None: ...

    def prepare_voice(self, ref_wav: str, ref_text: str) -> Any:
        """Per-speaker voice state (e.g. a cloning prompt), built once and reused."""
        ...

    def synthesize(
        self,
        text: str,
        voice: Any,
        target_duration: float | None = None,
        max_seconds: float | None = None,
    ) -> np.ndarray:
        """Mono float32 audio at `sample_rate`, never longer than `max_seconds`: autoregressive
        models can miss their end token and generate until a VRAM-filling limit."""
        ...

    def settings(self) -> dict[str, Any]:
        """Options that change the output (part of the per-segment cache key)."""
        ...


# Optional: `design_voice(description, text, seed) -> (audio, sample_rate)` lets a
# backend build the fallback voice bank (synthesize/voicebank.py).


BACKENDS = {
    "omnivoice": "synthesize.backends.omnivoice:OmniVoiceBackend",
}


def backend_class(name: str) -> Any:
    if name not in BACKENDS:
        raise ValueError(f"unknown TTS backend {name!r} (known: {', '.join(BACKENDS)})")
    module, cls = BACKENDS[name].split(":")
    return getattr(importlib.import_module(module), cls)


def create(name: str, options: dict[str, Any], *, language: str, models_dir: str) -> TTSBackend:
    return backend_class(name)(options, language=language, models_dir=models_dir)
