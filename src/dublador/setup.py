"""`dublador setup`: one uv venv per stage, outside the repository (ADR-0007, ADR-0014)."""

import hashlib
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from dublador.config import Config, write_snapshot
from dublador.runner import stage_env
from dublador.stages import ExtraWheel, StageSpec


class SetupError(Exception):
    pass


def installable(specs: Sequence[StageSpec], only: Sequence[str] = ()) -> list[StageSpec]:
    """Subprocess stages that have a project under stages/, optionally filtered."""
    known = {s.name for s in specs if s.module}
    unknown = sorted(set(only) - known)
    if unknown:
        raise SetupError(f"unknown stage(s): {', '.join(unknown)}")
    return [
        s
        for s in specs
        if s.module
        and (not only or s.name in only)
        and (s.project_dir / "pyproject.toml").is_file()
    ]


def _device(path: Path) -> int:
    """Filesystem of a path, or of its closest existing parent."""
    while not path.exists():
        path = path.parent
    return path.stat().st_dev


def same_filesystem(a: Path, b: Path) -> bool:
    return _device(a) == _device(b)


def uv_env(config: Config, stage: str) -> dict[str, str]:
    paths = config.paths
    # A VIRTUAL_ENV inherited from `uv run dublador` points at the orchestrator's venv.
    inherited = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    return dict(
        inherited,
        UV_PROJECT_ENVIRONMENT=str(paths.venvs_dir / stage),
        UV_CACHE_DIR=str(paths.uv_cache_dir),
        # Python builds uv downloads for a stage stay next to the venvs, not in ~.
        UV_PYTHON_INSTALL_DIR=str(paths.venvs_dir / "_python"),
        # Reflink on btrfs: venvs share disk blocks with the cache (falls back to copy).
        UV_LINK_MODE="clone",
    )


def _check(cmd: list[str], *, env: dict[str, str], cwd: Path, what: str) -> None:
    try:
        code = subprocess.run(cmd, env=env, cwd=cwd, check=False).returncode
    except FileNotFoundError as e:
        raise SetupError(f"{cmd[0]} not found") from e
    if code != 0:
        raise SetupError(f"{what} failed (exit {code})")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_wheel(extra: ExtraWheel, downloads: Path) -> Path:
    """Download the archive once (checked against its sha256) and extract the wheel."""
    folder = downloads / extra.sha256[:16]
    archive = folder / extra.url.rsplit("/", 1)[-1]
    wheel = folder / Path(extra.member).name
    if wheel.is_file():
        return wheel
    folder.mkdir(parents=True, exist_ok=True)
    if not archive.is_file() or _sha256(archive) != extra.sha256:
        partial = archive.with_name(archive.name + ".part")
        try:
            with urllib.request.urlopen(extra.url) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
        except OSError as e:
            raise SetupError(f"could not download {extra.url}: {e}") from e
        if _sha256(partial) != extra.sha256:
            partial.unlink()
            raise SetupError(f"{extra.url}: sha256 mismatch")
        partial.replace(archive)
    with zipfile.ZipFile(archive) as zf:
        try:
            data = zf.read(extra.member)
        except KeyError as e:
            raise SetupError(f"{archive.name} has no {extra.member}") from e
    partial = wheel.with_name(wheel.name + ".part")
    partial.write_bytes(data)
    partial.replace(wheel)
    return wheel


def link_wheel(wheel: Path, project_dir: Path) -> Path:
    """stages/<name>/wheels/<file> -> the wheel in models_dir (the folder is git-ignored)."""
    link = project_dir / "wheels" / wheel.name
    link.parent.mkdir(exist_ok=True)
    if not (link.is_symlink() and link.readlink() == wheel):
        tmp = link.with_name(f".{link.name}.tmp")
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(wheel)
        tmp.replace(link)
    return link


def run_setup(
    config: Config,
    specs: Sequence[StageSpec],
    console: Console,
    *,
    only: Sequence[str] = (),
    prefetch: bool = True,
) -> list[str]:
    """Create every stage venv and prefetch its models. Returns the stages set up."""
    paths = config.paths
    if config.source_file is None:
        console.print(
            "[yellow]warning:[/yellow] no config.toml found; using default paths under "
            f"{paths.venvs_dir.parent} (see config.example.toml)"
        )
    for path in (paths.work_dir, paths.models_dir, paths.venvs_dir, paths.uv_cache_dir):
        path.mkdir(parents=True, exist_ok=True)
    if not same_filesystem(paths.uv_cache_dir, paths.venvs_dir):
        console.print(
            "[yellow]warning:[/yellow] uv_cache_dir and venvs_dir are on different "
            "filesystems; every venv will hold a full copy of torch"
        )

    selected = installable(specs, only)
    if not selected:
        console.print("no stage projects to set up yet")
        return []

    with tempfile.TemporaryDirectory(prefix="dublador-setup-") as tmp:
        config_file = write_snapshot(config, Path(tmp))
        for spec in selected:
            console.rule(spec.name)
            env = uv_env(config, spec.name)
            for extra in spec.extra_wheels:
                link_wheel(fetch_wheel(extra, paths.models_dir / "_downloads"), spec.project_dir)
            _check(
                ["uv", "sync", "--project", str(spec.project_dir)],
                env=env,
                cwd=spec.project_dir,
                what=f"{spec.name}: uv sync",
            )
            if prefetch:
                assert spec.module is not None
                _check(
                    [
                        str(spec.python_path(config)),
                        "-m",
                        spec.module,
                        "prefetch",
                        "--config",
                        str(config_file),
                    ],
                    env=stage_env(config),
                    cwd=spec.project_dir,
                    what=f"{spec.name}: prefetch",
                )
            console.print(f"[green]✓[/green] {spec.name} → {paths.venvs_dir / spec.name}")
    return [s.name for s in selected]
