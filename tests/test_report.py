from pathlib import Path

from dublador.manifest import Manifest
from dublador.report import render, write_report


def make_manifest(tmp_path: Path) -> Manifest:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"v")
    m = Manifest.create(
        tmp_path / "manifest.json", job_id="v-1", video=video, fingerprint="sha256:" + "0" * 64,
        output_path=tmp_path / "v.ptbr.mkv", audio_track=None, stages=["extract", "mix", "mux"],
    )  # fmt: skip
    m.data["stages"]["extract"] = {
        "status": "done",
        "started_at": "2026-10-10T10:00:00Z",
        "finished_at": "2026-10-10T10:01:05Z",
        "warnings": [],
    }
    m.data["stages"]["mix"] = {
        "status": "failed",
        "started_at": "2026-10-10T10:01:05Z",
        "finished_at": "2026-10-10T10:01:09Z",
        "error": "mix failed | badly",
        "warnings": ["mix would clip; lowered by 2.1 dB"],
    }
    m.save()
    return m


def test_render(tmp_path: Path) -> None:
    text = render(make_manifest(tmp_path))
    assert "| extract | done | 1m05s | 0 |  |" in text
    assert "| mix | failed | 4s | 1 | mix failed \\| badly |" in text
    assert "| mux | pending | — | 0 |  |" in text
    assert "Time in stages (last run of each): 1m09s" in text
    assert "### mix\n\n- mix would clip; lowered by 2.1 dB" in text


def test_write_report(tmp_path: Path) -> None:
    path = write_report(make_manifest(tmp_path), tmp_path)
    assert path.name == "report.md" and path.read_text().startswith("# dublador job `v-1`")
