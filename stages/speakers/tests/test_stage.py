"""The stage on a diarized transcript, with a fake gender classifier."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile
from stagekit import main, read_json, write_json

SR = 16000


class FakeClassifier:
    """Female for clips louder than 0.3, male otherwise (speakers differ in level)."""

    def p_female(self, clip: np.ndarray) -> float:
        return 0.9 if float(np.max(np.abs(clip))) > 0.3 else 0.1


def segment(i: int, start: float, end: float, speaker: str) -> dict[str, Any]:
    words, t = [], start
    while t + 0.4 <= end:
        words.append({"word": "word", "start": round(t, 3), "end": round(t + 0.4, 3), "score": 0.9})
        t += 0.5
    return {"id": f"seg_{i:04d}", "start": start, "end": end, "text": "word " * len(words),
            "speaker": speaker, "words": words}  # fmt: skip


def run_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from speakers.__main__ import run

    monkeypatch.setattr("speakers.__main__.make_classifier", FakeClassifier)
    job = tmp_path / "job"
    (job / "02_separate").mkdir(parents=True)
    rng = np.random.default_rng(0)
    vocals = np.zeros(40 * SR, dtype=np.float32)
    vocals[: 20 * SR] = rng.normal(0, 0.05, 20 * SR)  # SPK_00: quiet, 20 s
    vocals[20 * SR : 21 * SR] = rng.normal(0, 0.2, SR)  # SPK_01: loud, 1 s (no reference)
    soundfile.write(str(job / "02_separate" / "vocals.wav"), vocals, SR)
    soundfile.write(str(job / "02_separate" / "accompaniment.wav"), vocals * 0, SR)
    segs = [segment(1, 0.0, 9.9, "SPK_00"), segment(2, 10.0, 19.9, "SPK_00"),
            segment(3, 20.0, 21.0, "SPK_01")]  # fmt: skip
    write_json(job / "04_diarize" / "transcript.json",
               {"schema": "transcript/v1", "language": "en", "segments": segs})  # fmt: skip
    config = tmp_path / "config.toml"
    config.write_text("[speakers]\ngender_clip_min_seconds = 0.8\n")
    main("speakers", run=run, argv=["--job-dir", str(job), "--config", str(config)])
    return read_json(job / "05_speakers" / "speakers.json", "speakers/v1")


def test_clone_with_reference_and_bank_without(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    doc = run_stage(tmp_path, monkeypatch)
    spk0, spk1 = doc["speakers"]
    assert spk0["strategy"] == "clone" and spk0["reference"] is not None
    assert spk0["gender"] == "male" and spk0["voice"] is None
    assert spk1["strategy"] == "bank" and spk1["reference"] is None
    assert spk1["gender"] == "female" and spk1["voice"] == "female_01"
