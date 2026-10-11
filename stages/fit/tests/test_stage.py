"""The stage with real ffmpeg (rubberband) on synthetic clips."""

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile
from stagekit import main, read_json, write_json

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
SR = 24000


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def make_job(tmp_path: Path, naturals: list[float]) -> tuple[Path, Path]:
    """Lines 1 s long that start every 3 s, with clips of the given lengths."""
    job = tmp_path / "job"
    lines, clips = [], []
    for i, natural in enumerate(naturals, start=1):
        seg_id = f"seg_{i:04d}"
        path = f"07_synthesize/segments/{seg_id}.wav"
        (job / path).parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(job / path), tone(natural), SR)
        start = (i - 1) * 3.0
        lines.append({"id": seg_id, "start": start, "end": start + 1.0, "speaker": "SPK_00",
                      "source_text": "x", "text": "uma frase", "syllables": 20, "budget": 15,
                      "input_hash": "sha256:" + "0" * 64})  # fmt: skip
        clips.append({"id": seg_id, "path": path, "duration": natural,
                      "input_hash": "sha256:" + f"{i:064d}"})  # fmt: skip
    write_json(job / "06_translate" / "translation.json", {
        "schema": "translation/v1", "model": "m", "source_language": "en",
        "target_language": "pt-BR", "segments": lines,
    })  # fmt: skip
    synthesis = {"schema": "synthesis/v1", "backend": "fake", "sample_rate": SR, "segments": clips}
    write_json(job / "07_synthesize" / "synthesis.json", synthesis)
    config = tmp_path / "config.toml"
    config.write_text("[fit]\nmax_speedup = 1.25\ngap_seconds = 0.15\nmax_forced_speedup = 1.6\n")
    return job, config


def run_stage(job: Path, config: Path, *extra: str) -> dict[str, Any]:
    from fit.__main__ import run

    main("fit", run=run, argv=["--job-dir", str(job), "--config", str(config), *extra])
    return {s["id"]: s for s in read_json(job / "08_fit" / "fit.json", "fit/v1")["segments"]}


def test_strategies_and_stretch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Room is 3.0 - 0.15 = 2.85 s for every line but the last.
    job, config = make_job(tmp_path, [0.8, 2.0, 3.3, 5.0, 1.0])
    fit = run_stage(job, config)
    assert [fit[f"seg_{i:04d}"]["strategy"] for i in range(1, 6)] == [
        "fits", "silence", "stretch", "overflow", "fits",
    ]  # fmt: skip
    stretched = soundfile.info(str(job / fit["seg_0003"]["path"])).duration
    assert stretched == pytest.approx(2.85, abs=0.05)
    slowed = soundfile.info(str(job / fit["seg_0001"]["path"])).duration
    assert fit["seg_0001"]["ratio"] == 0.85 and slowed == pytest.approx(0.8 / 0.85, abs=0.05)
    over = fit["seg_0004"]
    assert over["ratio"] == 1.6 and over["attempt"] == 1
    assert over["shorten_to"] == 11  # 20 syllables * 2.85 / 5.0
    assert over["target_duration"] is None
    out = capsys.readouterr().out
    assert '"warning"' in out and "seg_0004: 5.0s of speech for a 2.8s window" in out


def test_only_redoes_listed_ids_and_counts_attempts(tmp_path: Path) -> None:
    job, config = make_job(tmp_path, [0.8, 5.0, 0.5])  # the last line has no limit
    first = run_stage(job, config)
    # The loop re-synthesized seg_0002, still too long: second attempt asks for a duration.
    doc = read_json(job / "07_synthesize" / "synthesis.json", "synthesis/v1")
    doc["segments"][1]["input_hash"] = "sha256:" + "f" * 64
    write_json(job / "07_synthesize" / "synthesis.json", doc)
    second = run_stage(job, config, "--only", "seg_0002")
    assert second["seg_0001"] == first["seg_0001"]
    assert second["seg_0002"]["attempt"] == 2
    assert second["seg_0002"]["shorten_to"] is None
    assert second["seg_0002"]["target_duration"] == 2.85
