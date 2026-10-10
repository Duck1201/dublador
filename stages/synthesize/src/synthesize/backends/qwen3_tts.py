"""Qwen3-TTS-12Hz-1.7B-Base: zero-shot voice cloning (Apache-2.0, docs/MODELS.md §6)."""

from typing import Any

import numpy as np

DEFAULTS: dict[str, Any] = {
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "dtype": "bfloat16",
}
# Qwen3-TTS takes language names, not codes.
LANGUAGES = {"pt": "Portuguese", "en": "English", "es": "Spanish", "fr": "French", "de": "German"}


class Qwen3TTSBackend:
    name = "qwen3_tts"
    supports_target_duration = False

    def __init__(self, options: dict[str, Any], *, language: str, models_dir: str) -> None:
        unknown = sorted(set(options) - set(DEFAULTS))
        if unknown:
            raise ValueError(f"unknown qwen3_tts option(s): {', '.join(unknown)}")
        self.options = {**DEFAULTS, **options}
        code = language.split("-")[0].lower()
        if code not in LANGUAGES:
            raise ValueError(f"qwen3_tts does not support language {language!r}")
        self.language = LANGUAGES[code]
        self.sample_rate = 0
        self._model: Any = None

    def settings(self) -> dict[str, Any]:
        return {**self.options, "language": self.language}

    def load(self) -> None:
        import torch  # pyright: ignore[reportMissingImports]
        from qwen_tts import Qwen3TTSModel  # pyright: ignore[reportMissingImports]

        self._model = Qwen3TTSModel.from_pretrained(
            self.options["model"],
            device_map="cuda",
            dtype=getattr(torch, self.options["dtype"]),
            attn_implementation="sdpa",  # no flash-attn wheels for ROCm gfx11
        )

    def prepare_voice(self, ref_wav: str, ref_text: str) -> Any:
        return self._model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)

    def synthesize(self, text: str, voice: Any, target_duration: float | None = None) -> np.ndarray:
        wavs, sr = self._model.generate_voice_clone(
            text=text, language=self.language, voice_clone_prompt=voice
        )
        self.sample_rate = sr
        return np.asarray(wavs[0], dtype=np.float32)

    @staticmethod
    def prefetch(options: dict[str, Any]) -> None:
        from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

        snapshot_download({**DEFAULTS, **options}["model"])
