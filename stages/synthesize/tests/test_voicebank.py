"""Building the fallback voice bank with a fake voice design."""

from pathlib import Path

import numpy as np
from synthesize import voicebank


def test_build_designs_only_missing_voices(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    def design(description: str, text: str, seed: int) -> tuple[np.ndarray, int]:
        calls.append((description, seed))
        return np.zeros(2400, dtype=np.float32), 24000

    assert len(voicebank.missing(tmp_path)) == len(voicebank.VOICES)
    voicebank.build(tmp_path, design, log=lambda _: None)
    assert len(calls) == len(voicebank.VOICES) and voicebank.missing(tmp_path) == []
    wav, txt = voicebank.paths(tmp_path, "male_03")
    assert wav == tmp_path / "voices" / "pt-BR" / "male" / "03.wav" and wav.is_file()
    assert txt.read_text().strip() == voicebank.TEXT

    calls.clear()
    voicebank.build(tmp_path, design, log=lambda _: None)
    assert calls == []
