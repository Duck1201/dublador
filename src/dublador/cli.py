"""Command-line interface. See docs/ARCHITECTURE.md#cli."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table
from stagekit import ContractError

from dublador import __version__, stages
from dublador.config import Config, ConfigError, load_config, write_snapshot
from dublador.doctor import run_checks
from dublador.ffmpeg import FfmpegError
from dublador.jobs import Job, JobError, find_job, list_jobs
from dublador.manifest import Manifest
from dublador.runner import PipelineError, open_job, reopen_job, run_pipeline
from dublador.setup import SetupError, run_setup

app = typer.Typer(
    name="dublador",
    help="Dub English videos into Brazilian Portuguese, fully offline.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


class OutputFormat(StrEnum):
    MKV = "mkv"
    MP4 = "mp4"


def _not_implemented(command: str) -> NoReturn:
    err_console.print(f"[red]error:[/red] `dublador {command}` is not implemented yet")
    raise typer.Exit(code=1)


def _load_config(
    overrides: dict[str, object] | None = None, *, snapshot: Path | None = None
) -> Config:
    env = dict(os.environ, DUBLADOR_CONFIG=str(snapshot)) if snapshot else None
    try:
        return load_config(overrides, env=env)
    except ConfigError as e:
        err_console.print(f"[red]config error:[/red] {e}")
        raise typer.Exit(code=2) from e


@contextmanager
def _errors() -> Iterator[None]:
    """Turn expected failures into a one-line error and exit code 1."""
    try:
        yield
    except (JobError, PipelineError, FfmpegError, ContractError, SetupError) as e:
        err_console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e
    except KeyboardInterrupt as e:
        err_console.print("[yellow]interrupted[/yellow]; continue with `dublador resume`")
        raise typer.Exit(code=130) from e


def _run(job: Job, manifest: Manifest, config: Config) -> None:
    write_snapshot(config, job.dir)
    console.print(f"job [bold]{job.id}[/bold] in {job.dir}")
    run_pipeline(job, manifest, config, stages.PIPELINE, console)
    console.print(f"[green]done:[/green] {manifest.output_path}")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"dublador {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    pass


@app.command()
def run(
    video: Annotated[Path | None, typer.Argument(help="Video to dub; omit to pick one.")] = None,
    source_lang: Annotated[str | None, typer.Option(help="Source language.")] = None,
    target_lang: Annotated[str | None, typer.Option(help="Target language.")] = None,
    translator_model: Annotated[str | None, typer.Option(help="Ollama model tag.")] = None,
    review: Annotated[bool, typer.Option(help="Pause after translation for review.")] = False,
    format: Annotated[OutputFormat | None, typer.Option(help="Output container.")] = None,
    output: Annotated[Path | None, typer.Option(help="Output file path.")] = None,
    audio_track: Annotated[int | None, typer.Option(help="Audio track index to dub.")] = None,
) -> None:
    """Dub a video from start to finish."""
    if review:
        _not_implemented("run --review")
    if video is None:
        err_console.print(
            "[red]error:[/red] missing VIDEO (the file picker is not implemented yet)"
        )
        raise typer.Exit(code=2)
    config = _load_config(
        {
            "languages.source": source_lang,
            "languages.target": target_lang,
            "translate.model": translator_model,
            "output.format": format.value if format else None,
        }
    )
    with _errors():
        job, manifest = open_job(
            video.expanduser().resolve(),
            config,
            stages.PIPELINE,
            output=output.expanduser().resolve() if output else None,
            audio_track=audio_track,
        )
        _run(job, manifest, config)


@app.command()
def resume(
    target: Annotated[str | None, typer.Argument(help="Job ID or video path.")] = None,
) -> None:
    """Resume an interrupted or reviewed job (default: the latest one)."""
    with _errors():
        job = find_job(target, _load_config().paths.work_dir)
        manifest = reopen_job(job)
        _run(job, manifest, _load_config(snapshot=job.snapshot_path))


@app.command()
def status(
    job_id: Annotated[str | None, typer.Argument(help="Job ID.")] = None,
) -> None:
    """Show the stages of a job, or list all jobs."""
    work_dir = _load_config().paths.work_dir
    with _errors():
        if job_id is None:
            _print_jobs(list_jobs(work_dir), work_dir)
        else:
            _print_stages(Manifest.load(find_job(job_id, work_dir).manifest_path))


_STATUS_STYLE = {"done": "green", "failed": "red", "running": "yellow", "pending": "dim"}


def _job_state(manifest: Manifest) -> str:
    statuses = [entry["status"] for entry in manifest.data["stages"].values()]
    for state in ("failed", "running"):
        if state in statuses:
            return state
    return "done" if all(s == "done" for s in statuses) else "pending"


def _print_jobs(jobs: list[Job], work_dir: Path) -> None:
    if not jobs:
        console.print(f"no jobs in {work_dir}")
        return
    table = Table("job", "video", "stages", "state", "updated")
    for job in jobs:
        manifest = Manifest.load(job.manifest_path)
        entries = manifest.data["stages"].values()
        done = sum(entry["status"] == "done" for entry in entries)
        state = _job_state(manifest)
        updated = datetime.fromtimestamp(job.manifest_path.stat().st_mtime)
        table.add_row(
            job.id,
            manifest.video_path.name,
            f"{done}/{len(entries)}",
            f"[{_STATUS_STYLE[state]}]{state}[/]",
            updated.strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


def _print_stages(manifest: Manifest) -> None:
    console.print(f"job [bold]{manifest.job_id}[/bold]: {manifest.video_path}")
    console.print(f"output: {manifest.output_path}")
    table = Table("stage", "status", "started", "finished", "error")
    for name, entry in manifest.data["stages"].items():
        status = entry["status"]
        table.add_row(
            name,
            f"[{_STATUS_STYLE[status]}]{status}[/]",
            entry.get("started_at") or "",
            entry.get("finished_at") or "",
            entry.get("error") or "",
        )
    console.print(table)


@app.command()
def clean(
    job_id: Annotated[str | None, typer.Argument(help="Job ID.")] = None,
    done: Annotated[bool, typer.Option("--done", help="Clean every finished job.")] = False,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    """Remove heavy intermediate files from jobs."""
    _not_implemented("clean")


@app.command()
def setup(
    only: Annotated[
        list[str] | None, typer.Option("--only", help="Set up only this stage (repeatable).")
    ] = None,
    skip_prefetch: Annotated[
        bool, typer.Option("--skip-prefetch", help="Create venvs without downloading models.")
    ] = False,
) -> None:
    """Create the stage venvs and download the models."""
    config = _load_config()
    with _errors():
        run_setup(config, stages.PIPELINE, console, only=only or (), prefetch=not skip_prefetch)


_LEVEL_STYLE = {"ok": "[green]ok[/]", "warn": "[yellow]warn[/]", "error": "[red]error[/]"}


@app.command()
def doctor(
    min_free_gb: Annotated[float, typer.Option(help="Error below this much free disk.")] = 10,
    warn_free_gb: Annotated[float, typer.Option(help="Warn below this much free disk.")] = 50,
) -> None:
    """Check GPU, ffmpeg, Ollama, models and disk space."""
    config = _load_config()
    checks = run_checks(config, stages.PIPELINE, min_free_gb=min_free_gb, warn_free_gb=warn_free_gb)
    table = Table("check", "status", "detail")
    for check in checks:
        table.add_row(check.name, _LEVEL_STYLE[check.level], check.detail)
    console.print(table)
    if any(check.level == "error" for check in checks):
        raise typer.Exit(code=1)


def main() -> None:
    app()
