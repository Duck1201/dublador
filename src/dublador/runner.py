"""Runs the pipeline: one stage at a time, skipping stages whose inputs did not change."""

import json
import os
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from stagekit import hash_file, hash_value

from dublador.config import Config
from dublador.jobs import Job, JobError, job_for_video, video_fingerprint
from dublador.manifest import Manifest
from dublador.stages import StageSpec

_RUN_MARKER = "=== dublador run"


class PipelineError(Exception):
    """A stage failed; details are in the manifest and the stage log."""


def stage_env(config: Config, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for stage processes: model caches inside models_dir (ADR-0014)."""
    models = config.paths.models_dir
    env = dict(os.environ)
    env.update(
        HF_HOME=str(models / "hf"),
        TORCH_HOME=str(models / "torch"),
        MIOPEN_USER_DB_PATH=str(models / "miopen"),
    )
    # MIOpen's default exhaustive search compiles convolution kernels for every new
    # input shape; with variable-length audio that made TTS 12-30x slower (MODELS.md).
    env.setdefault("MIOPEN_FIND_MODE", "FAST")
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    if config.gpu.hsa_override_gfx_version:
        env["HSA_OVERRIDE_GFX_VERSION"] = config.gpu.hsa_override_gfx_version
    env.update(extra or {})
    return env


def inputs_hash(spec: StageSpec, job: Job, manifest: Manifest, config: Config) -> str:
    settings = config.to_toml_dict()
    parts: dict[str, object] = {
        "version": spec.resolved_version(),
        "config": {s: settings.get(s) for s in (spec.name, *spec.config_sections)},
        "options": {k: manifest.data["options"][k] for k in spec.options},
        "files": {},
    }
    if spec.uses_video:
        parts["video"] = manifest.video_hash
    files: dict[str, str | None] = {}
    for rel in spec.inputs:
        path = job.dir / rel
        files[rel] = hash_file(path) if path.is_file() else None
    parts["files"] = files
    return hash_value(parts)


def _outputs_present(spec: StageSpec, job: Job, manifest: Manifest) -> bool:
    if spec.writes_output_video and not manifest.output_path.is_file():
        return False
    return all((job.dir / rel).exists() for rel in spec.outputs)


def run_pipeline(
    job: Job,
    manifest: Manifest,
    config: Config,
    specs: Sequence[StageSpec],
    console: Console,
) -> None:
    for spec in specs:
        digest = inputs_hash(spec, job, manifest, config)
        if manifest.is_current(spec.name, digest) and _outputs_present(spec, job, manifest):
            console.print(f"[dim]✓ {spec.name} (up to date)[/dim]")
            continue
        missing = [rel for rel in spec.inputs if not (job.dir / rel).exists()]
        if missing:
            raise PipelineError(f"{spec.name}: missing inputs: {', '.join(missing)}")

        manifest.mark_running(spec.name, inputs_hash=digest, version=spec.resolved_version())
        try:
            if spec.function is not None:
                with console.status(f"{spec.name}…"):
                    spec.function(job.dir, manifest, config)
            else:
                _run_subprocess(spec, job, config, console)
            if not _outputs_present(spec, job, manifest):
                raise PipelineError(f"{spec.name}: finished without writing its outputs")
        except KeyboardInterrupt:
            manifest.mark_failed(spec.name, "interrupted")
            raise
        except Exception as e:
            message = str(e) if isinstance(e, PipelineError) else f"{spec.name}: {e}"
            manifest.mark_failed(spec.name, message)
            raise PipelineError(message) from e
        manifest.mark_done(spec.name)
        console.print(f"[green]✓[/green] {spec.name}")


def _run_subprocess(spec: StageSpec, job: Job, config: Config, console: Console) -> None:
    assert spec.module is not None
    python = spec.python_path(config)
    if not python.exists():
        raise PipelineError(
            f"{spec.name}: venv not found at {python.parent.parent}; run `dublador setup`"
        )
    cmd = [
        str(python),
        "-m",
        spec.module,
        "--job-dir",
        str(job.dir),
        "--config",
        str(job.snapshot_path),
    ]
    log_path = job.log_path(spec.name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    columns = (
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with log_path.open("a") as log, Progress(*columns, console=console, transient=True) as bar:
        log.write(f"{_RUN_MARKER} {datetime.now().isoformat(timespec='seconds')}\n")
        log.flush()
        task = bar.add_task(spec.name, total=None)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            env=stage_env(config, dict(spec.env)),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _handle_line(line, log, bar, task, console)
        code = proc.wait()

    if code != 0:
        raise PipelineError(f"{spec.name} failed (exit {code}): {_last_error(log_path)}")


def _handle_line(line: str, log: TextIO, bar: Progress, task: TaskID, console: Console) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        event = None
    if not isinstance(event, dict):
        # Libraries print to stdout too; keep their output in the log only.
        log.write(line)
        log.flush()
        return
    kind = event.get("event")
    if kind == "progress":
        bar.update(task, completed=event.get("done", 0), total=event.get("total"))
    elif kind == "warning":
        console.print(f"[yellow]warning:[/yellow] {event.get('message', '')}")
    elif kind == "log":
        console.print(f"[dim]{event.get('message', '')}[/dim]")


def _last_error(log_path: Path) -> str:
    text = log_path.read_text(errors="replace")
    current_run = text.rsplit(_RUN_MARKER, 1)[-1].splitlines()[1:]
    lines = [ln.strip() for ln in current_run if ln.strip()]
    for line in reversed(lines):
        if line.startswith("error:"):
            return line.removeprefix("error:").strip()
    return lines[-1] if lines else f"no output; see {log_path}"


def default_output(video: Path, config: Config) -> Path:
    suffix = config.languages.target.lower().replace("-", "")
    return video.with_name(f"{video.stem}.{suffix}.{config.output.format}")


def open_job(
    video: Path,
    config: Config,
    specs: Sequence[StageSpec],
    *,
    output: Path | None,
    audio_track: int | None,
) -> tuple[Job, Manifest]:
    """Create the job of a video, or reopen it with this run's options."""
    job, fingerprint = job_for_video(video, config.paths.work_dir)
    output = output or default_output(video, config)
    if output.resolve() == video.resolve():
        raise JobError("the output path must differ from the input video")
    job.dir.mkdir(parents=True, exist_ok=True)
    if job.exists():
        manifest = Manifest.load(job.manifest_path)
        manifest.set_options(output_path=output, audio_track=audio_track)
        manifest.save()
    else:
        manifest = Manifest.create(
            job.manifest_path,
            job_id=job.id,
            video=video,
            fingerprint=fingerprint,
            output_path=output,
            audio_track=audio_track,
            stages=[s.name for s in specs],
        )
    return job, manifest


def reopen_job(job: Job) -> Manifest:
    """Load a job for `resume`, checking that its video is still the same file."""
    manifest = Manifest.load(job.manifest_path)
    video = manifest.video_path
    if not video.is_file():
        raise JobError(f"the job's video is gone: {video}")
    if video_fingerprint(video) != manifest.video_hash:
        raise JobError(f"{video} changed since the job started; run `dublador run` on it again")
    return manifest
