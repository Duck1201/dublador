"""OmniVoice (k2-fsa): zero-shot voice cloning in 600+ languages, non-autoregressive
(code Apache-2.0, weights CC-BY-NC; docs/MODELS.md §6, ADR-0017)."""

from typing import Any

import numpy as np

DEFAULTS: dict[str, Any] = {
    "model": "k2-fsa/OmniVoice",
    "revision": "c5fdb5ccb189668d56333f77ba2629f4cd7535f4",
    "dtype": "float16",
    "num_step": 32,
    # Higher values made the speech glitchy in the bake-off (docs/MODELS.md §6).
    "guidance_scale": 2.0,
}


class OmniVoiceBackend:
    name = "omnivoice"
    supports_target_duration = True

    def __init__(self, options: dict[str, Any], *, language: str, models_dir: str) -> None:
        unknown = sorted(set(options) - set(DEFAULTS))
        if unknown:
            raise ValueError(f"unknown omnivoice option(s): {', '.join(unknown)}")
        self.options = {**DEFAULTS, **options}
        self.language = language.split("-")[0].lower()
        self.sample_rate = 0
        self._model: Any = None

    def settings(self) -> dict[str, Any]:
        return {**self.options, "language": self.language}

    def load(self) -> None:
        import torch  # pyright: ignore[reportMissingImports]
        from omnivoice import OmniVoice  # pyright: ignore[reportMissingImports]

        self._model = OmniVoice.from_pretrained(
            _download(self.options),
            device_map="cuda:0",
            dtype=getattr(torch, self.options["dtype"]),
        )
        self.sample_rate = self._model.sampling_rate

    def prepare_voice(self, ref_wav: str, ref_text: str) -> Any:
        return self._model.create_voice_clone_prompt(ref_audio=ref_wav, ref_text=ref_text)

    def synthesize(
        self,
        text: str,
        voice: Any,
        target_duration: float | None = None,
        max_seconds: float | None = None,
    ) -> np.ndarray:
        import torch  # pyright: ignore[reportMissingImports]

        try:
            audio = self._model.generate(
                text=text,
                language=self.language,
                voice_clone_prompt=voice,
                duration=target_duration,
                num_step=self.options["num_step"],
                guidance_scale=self.options["guidance_scale"],
            )[0]
        finally:
            torch.cuda.empty_cache()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        # The length is planned from the text up front, so it cannot run away like an
        # autoregressive model; the budget is still honored.
        if max_seconds is not None:
            audio = audio[: int(max_seconds * self.sample_rate)]
        return audio

    @staticmethod
    def prefetch(options: dict[str, Any]) -> None:
        _download({**DEFAULTS, **options})


def _download(options: dict[str, Any]) -> str:
    from huggingface_hub import snapshot_download  # pyright: ignore[reportMissingImports]

    return snapshot_download(options["model"], revision=options["revision"])
