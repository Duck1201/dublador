"""The stage with a fake TTS backend: caching, --only, checkpoints, errors."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile
from stagekit import main, read_json, write_json
from synthesize import backends
from synthesize.__main__ import run

CALLS: list[str] = []


class FakeBackend:
    name = "fake"
    supports_target_duration = False

    def __init__(self, options: dict[str, Any], *, language: str, models_dir: str) -> None:
        self.options = options
        self.sample_rate = 0

    def settings(self) -> dict[str, Any]:
        return dict(self.options)

    def load(self) -> None:
        CALLS.append("load")

    def prepare_voice(self, ref_wav: str, ref_text: str) -> str:
        CALLS.append(f"voice:{Path(ref_wav).name}")
        return ref_text

    def synthesize(
        self,
        text: str,
        voice: Any,
        target_duration: float | None = None,
        max_seconds: float | None = None,
    ) -> np.ndarray:
        CALLS.append(f"say:{text}")
        self.sample_rate = 24000
        # "loop" never reaches its end token; "flaky" misses it only the first time.
        runaway = text == "loop" or (text == "flaky" and CALLS.count("say:flaky") == 1)
        seconds = max_seconds if runaway and max_seconds else 0.1 * len(text)
        return np.zeros(int(seconds * 24000), dtype=np.float32)


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    CALLS.clear()
    monkeypatch.setitem(backends.BACKENDS, "fake", f"{__name__}:FakeBackend")
    monkeypatch.setattr("synthesize.__main__.wait_for_vram", lambda *a, **k: None)


def make_job(tmp_path: Path, texts: list[str], *, extra: str = "") -> tuple[Path, Path]:
    job = tmp_path / "job"
    ref = job / "05_speakers" / "refs" / "SPK_00.wav"
    ref.parent.mkdir(parents=True)
    soundfile.write(str(ref), np.zeros(24000, dtype=np.float32), 24000)
    write_json(job / "05_speakers" / "speakers.json", {"schema": "speakers/v1", "speakers": [{
        "id": "SPK_00", "speech_seconds": 9.0, "gender": "unknown", "gender_confidence": None,
        "strategy": "clone", "voice": None,
        "reference": {"path": "05_speakers/refs/SPK_00.wav", "start": 0, "end": 1, "text": "ref"},
    }]})  # fmt: skip
    segments = [
        {"id": f"seg_{i:04d}", "start": i * 2.0, "end": i * 2.0 + 1.5, "speaker": "SPK_00",
         "source_text": "x", "text": t, "input_hash": "sha256:" + "0" * 64}
        for i, t in enumerate(texts, start=1)
    ]  # fmt: skip
    write_json(job / "06_translate" / "translation.json", {
        "schema": "translation/v1", "model": "m", "source_language": "en",
        "target_language": "pt-BR", "segments": segments,
    })  # fmt: skip
    config = tmp_path / "config.toml"
    config.write_text(f'[paths]\nmodels_dir = "/m"\n[synthesize]\nbackend = "fake"\n{extra}')
    return job, config


def run_stage(job: Path, config: Path, *extra: str) -> dict[str, Any]:
    main("synthesize", run=run, argv=["--job-dir", str(job), "--config", str(config), *extra])
    return read_json(job / "07_synthesize" / "synthesis.json", "synthesis/v1")


def test_synthesizes_every_segment(tmp_path: Path) -> None:
    job, config = make_job(tmp_path, ["Olá", "Tudo bem?"])
    doc = run_stage(job, config)
    assert doc["sample_rate"] == 24000 and doc["backend"] == "fake"
    assert [s["duration"] for s in doc["segments"]] == [0.3, 0.9]
    assert (job / doc["segments"][1]["path"]).is_file()
    assert CALLS == ["load", "voice:SPK_00.wav", "say:Olá", "say:Tudo bem?"]


def test_cache_only_and_changed_text(tmp_path: Path) -> None:
    job, config = make_job(tmp_path, ["Olá", "Tudo bem?"])
    run_stage(job, config)
    CALLS.clear()
    run_stage(job, config)
    assert CALLS == []  # everything cached; the model is not even loaded

    run_stage(job, config, "--only", "seg_0002")
    assert CALLS[-1] == "say:Tudo bem?" and "say:Olá" not in CALLS

    CALLS.clear()
    job2, _ = make_job(tmp_path / "b", ["Olá", "Mudou"])
    (job2 / "07_synthesize").mkdir()
    for f in ("synthesis.json",):
        (job2 / "07_synthesize" / f).write_bytes((job / "07_synthesize" / f).read_bytes())
    (job2 / "07_synthesize" / "segments").mkdir()
    for wav in (job / "07_synthesize" / "segments").iterdir():
        (job2 / "07_synthesize" / "segments" / wav.name).write_bytes(wav.read_bytes())
    run_stage(job2, config)
    assert [c for c in CALLS if c.startswith("say:")] == ["say:Mudou"]


def test_missing_wav_is_resynthesized(tmp_path: Path) -> None:
    job, config = make_job(tmp_path, ["Olá", "Tudo bem?"])
    run_stage(job, config)
    (job / "07_synthesize" / "segments" / "seg_0001.wav").unlink()
    CALLS.clear()
    run_stage(job, config)
    assert [c for c in CALLS if c.startswith("say:")] == ["say:Olá"]


def test_backend_options_are_part_of_the_cache_key(tmp_path: Path) -> None:
    job, config = make_job(tmp_path, ["Olá"])
    run_stage(job, config)
    CALLS.clear()
    config.write_text(config.read_text() + 'voice_style = "calm"\n')
    run_stage(job, config)
    assert "say:Olá" in CALLS


def test_runaway_generation_is_retried_then_kept_with_a_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job, config = make_job(tmp_path, ["flaky", "loop"])
    doc = run_stage(job, config)
    assert [c for c in CALLS if c.startswith("say:")] == ["say:flaky"] * 2 + ["say:loop"] * 2
    # Each segment's window is 1.5 s, so the budget is 3 x 1.5 s.
    assert [s["duration"] for s in doc["segments"]] == [0.5, 4.5]
    out = capsys.readouterr().out
    assert out.count('"warning"') == 1
    assert "seg_0002: fake did not finish the sentence within 4.5 s twice" in out


def test_unknown_backend(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    job, config = make_job(tmp_path, ["Olá"])
    config.write_text(config.read_text().replace('"fake"', '"nope"'))
    with pytest.raises(SystemExit):
        run_stage(job, config)
    assert "unknown TTS backend 'nope'" in capsys.readouterr().err
