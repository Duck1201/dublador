"""Command-line interface. See docs/ARCHITECTURE.md#cli."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from dublador import __version__

app = typer.Typer(
    name="dublador",
    help="Dub English videos into Brazilian Portuguese, fully offline.",
    no_args_is_help=True,
    add_completion=False,
)
err_console = Console(stderr=True)


class OutputFormat(StrEnum):
    MKV = "mkv"
    MP4 = "mp4"


def _not_implemented(command: str) -> NoReturn:
    err_console.print(f"[red]error:[/red] `dublador {command}` is not implemented yet")
    raise typer.Exit(code=1)


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
    _not_implemented("run")


@app.command()
def resume(
    target: Annotated[str | None, typer.Argument(help="Job ID or video path.")] = None,
) -> None:
    """Resume an interrupted or reviewed job."""
    _not_implemented("resume")


@app.command()
def status(
    job_id: Annotated[str | None, typer.Argument(help="Job ID.")] = None,
) -> None:
    """Show the state of a job."""
    _not_implemented("status")


@app.command()
def clean(
    job_id: Annotated[str | None, typer.Argument(help="Job ID.")] = None,
    done: Annotated[bool, typer.Option("--done", help="Clean every finished job.")] = False,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    """Remove heavy intermediate files from jobs."""
    _not_implemented("clean")


@app.command()
def setup() -> None:
    """Create the stage venvs and download the models."""
    _not_implemented("setup")


@app.command()
def doctor() -> None:
    """Check GPU, ffmpeg, Ollama, models and disk space."""
    _not_implemented("doctor")


def main() -> None:
    app()
