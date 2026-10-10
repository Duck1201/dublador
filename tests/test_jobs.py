from pathlib import Path

import pytest

from dublador.jobs import JobError, find_job, job_for_video, slugify, video_fingerprint


def test_slugify() -> None:
    assert slugify("Palestra Técnica (Parte 1)") == "palestra-tecnica-parte-1"
    assert slugify("???") == "video"
    assert len(slugify("x" * 100)) == 40


def test_job_id_is_stable_and_content_based(tmp_path: Path) -> None:
    a = tmp_path / "My Video.mp4"
    a.write_bytes(b"a" * 1000)
    job, fingerprint = job_for_video(a, tmp_path / "work")
    assert job.id.startswith("my-video-")
    assert job.dir == tmp_path / "work" / job.id
    assert job_for_video(a, tmp_path / "work")[0] == job
    a.write_bytes(b"b" * 1000)
    assert video_fingerprint(a) != fingerprint


def test_fingerprint_samples_large_files(tmp_path: Path) -> None:
    big = tmp_path / "big.bin"
    with big.open("wb") as f:
        f.truncate(20 << 20)
    before = video_fingerprint(big)
    with big.open("r+b") as f:
        f.seek(10 << 20)
        f.write(b"changed")
    assert video_fingerprint(big) != before


def test_find_job_errors(tmp_path: Path) -> None:
    with pytest.raises(JobError, match="no jobs"):
        find_job(None, tmp_path)
    with pytest.raises(JobError, match="no job or video"):
        find_job("nope", tmp_path)
    video = tmp_path / "v.mp4"
    video.write_bytes(b"v")
    with pytest.raises(JobError, match="dublador run"):
        find_job(str(video), tmp_path)


def test_missing_video(tmp_path: Path) -> None:
    with pytest.raises(JobError, match="video not found"):
        job_for_video(tmp_path / "nope.mp4", tmp_path)
