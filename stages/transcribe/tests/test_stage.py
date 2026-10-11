"""The stage with a fake WhisperX: GPU first, CPU when the GPU fails."""

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile
from stagekit import main, read_json

LOADED: list[tuple[str, str]] = []


def fake_whisperx(fail_on: str | None) -> types.ModuleType:
    mod = types.ModuleType("whisperx")

    class Model:
        def __init__(self, device: str) -> None:
            self.device = device

        def transcribe(self, audio: Any, **kwargs: Any) -> dict[str, Any]:
            if self.device == fail_on:
                raise RuntimeError("hipErrorIllegalAddress")
            return {"segments": [{"start": 0.5, "end": 1.5, "text": "Hello there."}]}

    def load_model(name: str, device: str, **kwargs: Any) -> Model:
        assert kwargs["vad_model"] == "cpu-vad"  # the VAD never follows Whisper to the GPU
        LOADED.append((device, kwargs["compute_type"]))
        return Model(device)

    def align(segments: list[dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
        words = [
            {"word": "Hello", "start": 0.5, "end": 0.9, "score": 0.9},
            {"word": "there.", "start": 1.0, "end": 1.5, "score": 0.9},
        ]
        return {"segments": [{**segments[0], "words": words}]}

    mod.load_audio = lambda path, sr: np.full(2 * sr, 0.1, dtype=np.float32)  # type: ignore[attr-defined]
    mod.load_model = load_model  # type: ignore[attr-defined]
    mod.load_align_model = lambda **kwargs: (object(), {})  # type: ignore[attr-defined]
    mod.align = align  # type: ignore[attr-defined]
    return mod


def run_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_on: str | None
) -> dict[str, Any]:
    from transcribe.__main__ import run

    LOADED.clear()
    monkeypatch.setattr("transcribe.__main__._cpu_vad", lambda: "cpu-vad")
    monkeypatch.setitem(sys.modules, "whisperx", fake_whisperx(fail_on))
    monkeypatch.setattr("transcribe.__main__._free_gpu", lambda: None)
    job = tmp_path / "job"
    vocals = job / "02_separate" / "vocals.wav"
    vocals.parent.mkdir(parents=True)
    soundfile.write(str(vocals), np.zeros(16000, dtype=np.float32), 16000)
    config = tmp_path / "config.toml"
    config.write_text('[languages]\nsource = "en"\n')
    main("transcribe", run=run, argv=["--job-dir", str(job), "--config", str(config)])
    return read_json(job / "03_transcribe" / "transcript.json", "transcript/v1")


def test_gpu_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = run_stage(tmp_path, monkeypatch, fail_on=None)
    assert LOADED == [("cuda", "int8_float16")]
    assert doc["segments"][0]["text"] == "Hello there."


def test_falls_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = run_stage(tmp_path, monkeypatch, fail_on="cuda")
    assert LOADED == [("cuda", "int8_float16"), ("cpu", "int8")]
    assert doc["segments"][0]["text"] == "Hello there."
    assert "retrying on the CPU (int8)" in capsys.readouterr().out
