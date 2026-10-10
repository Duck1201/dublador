"""Job identity and on-disk layout: <work_dir>/<job_id>/ (docs/ARCHITECTURE.md)."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from dublador.config import SNAPSHOT_NAME

MANIFEST_NAME = "manifest.json"
LOGS_DIR = "logs"

# Sampled content hash: hashing every byte of a multi-GB video on each run is
# slow, and size + head/middle/tail is enough to tell videos apart.
_SAMPLE = 4 << 20
_SLUG_MAX = 40


class JobError(Exception):
    """Job not found or not usable."""


def video_fingerprint(video: Path) -> str:
    size = video.stat().st_size
    digest = hashlib.sha256(str(size).encode())
    with video.open("rb") as f:
        for offset in sorted({0, max(0, size // 2 - _SAMPLE // 2), max(0, size - _SAMPLE)}):
            f.seek(offset)
            digest.update(f.read(_SAMPLE))
    return "sha256:" + digest.hexdigest()


def slugify(text: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:_SLUG_MAX].rstrip("-") or "video"


def job_id_for(video: Path, fingerprint: str) -> str:
    return f"{slugify(video.stem)}-{fingerprint.removeprefix('sha256:')[:8]}"


@dataclass(frozen=True)
class Job:
    id: str
    dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST_NAME

    @property
    def snapshot_path(self) -> Path:
        return self.dir / SNAPSHOT_NAME

    def log_path(self, stage: str) -> Path:
        return self.dir / LOGS_DIR / f"{stage}.log"

    def exists(self) -> bool:
        return self.manifest_path.is_file()


def job_for_video(video: Path, work_dir: Path) -> tuple[Job, str]:
    """The job of a video (it may not exist yet) and the video's fingerprint."""
    if not video.is_file():
        raise JobError(f"video not found: {video}")
    fingerprint = video_fingerprint(video)
    job_id = job_id_for(video, fingerprint)
    return Job(job_id, work_dir / job_id), fingerprint


def list_jobs(work_dir: Path) -> list[Job]:
    """Existing jobs, most recently updated first."""
    if not work_dir.is_dir():
        return []
    jobs = [Job(d.name, d) for d in work_dir.iterdir() if (d / MANIFEST_NAME).is_file()]
    return sorted(jobs, key=lambda j: j.manifest_path.stat().st_mtime, reverse=True)


def find_job(target: str | None, work_dir: Path) -> Job:
    """Resolve a job from a job id, a video path, or (None) the latest job."""
    if target is None:
        jobs = list_jobs(work_dir)
        if not jobs:
            raise JobError(f"no jobs in {work_dir}")
        return jobs[0]
    by_id = Job(target, work_dir / target)
    if "/" not in target and by_id.exists():
        return by_id
    path = Path(target).expanduser()
    if path.is_file():
        job, _ = job_for_video(path, work_dir)
        if job.exists():
            return job
        raise JobError(f"no job for {path}; start one with `dublador run`")
    raise JobError(f"no job or video named {target!r}")
