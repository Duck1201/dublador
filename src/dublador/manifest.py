"""manifest.json: per-stage state of a job, used to skip finished work on resume."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stagekit import read_json, write_json

SCHEMA = "manifest/v1"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class Manifest:
    path: Path
    data: dict[str, Any]

    @classmethod
    def create(
        cls,
        path: Path,
        *,
        job_id: str,
        video: Path,
        fingerprint: str,
        output_path: Path,
        audio_track: int | None,
        stages: Iterable[str],
    ) -> "Manifest":
        data = {
            "schema": SCHEMA,
            "job_id": job_id,
            "video": {"path": str(video), "hash": fingerprint, "size": video.stat().st_size},
            "options": {"output_path": str(output_path), "audio_track": audio_track},
            "created_at": _now(),
            "stages": {name: {"status": "pending"} for name in stages},
        }
        manifest = cls(path, data)
        manifest.save()
        return manifest

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        return cls(path, read_json(path, SCHEMA))

    def save(self) -> None:
        write_json(self.path, self.data)

    @property
    def job_id(self) -> str:
        return self.data["job_id"]

    @property
    def video_path(self) -> Path:
        return Path(self.data["video"]["path"])

    @property
    def video_hash(self) -> str:
        return self.data["video"]["hash"]

    @property
    def output_path(self) -> Path:
        return Path(self.data["options"]["output_path"])

    @property
    def audio_track(self) -> int | None:
        return self.data["options"]["audio_track"]

    def set_options(self, *, output_path: Path, audio_track: int | None) -> None:
        self.data["options"] = {"output_path": str(output_path), "audio_track": audio_track}

    def stage(self, name: str) -> dict[str, Any]:
        return self.data["stages"].setdefault(name, {"status": "pending"})

    def is_current(self, name: str, inputs_hash: str) -> bool:
        """Done with the same inputs: the stage can be skipped."""
        entry = self.stage(name)
        return entry["status"] == "done" and entry.get("inputs_hash") == inputs_hash

    def mark_running(self, name: str, *, inputs_hash: str, version: str | None) -> None:
        self.data["stages"][name] = {
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "version": version,
            "model": None,
            "inputs_hash": inputs_hash,
            "error": None,
        }
        self.save()

    def mark_done(self, name: str, *, model: str | None = None) -> None:
        entry = self.stage(name)
        entry.update(status="done", finished_at=_now(), model=model, error=None)
        self.save()

    def mark_failed(self, name: str, error: str) -> None:
        entry = self.stage(name)
        entry.update(status="failed", finished_at=_now(), error=error)
        self.save()
