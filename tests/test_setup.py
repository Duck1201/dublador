import io
import shutil
import textwrap
from pathlib import Path

import pytest
from rich.console import Console

from dublador.config import Config, load_config
from dublador.setup import SetupError, installable, run_setup, same_filesystem
from dublador.stages import PIPELINE, StageSpec

needs_uv = pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")

# A stage without dependencies, so `uv sync` works offline. Its prefetch
# records where the orchestrator told it to keep models.
TOY_MAIN = """
import os, sys
from pathlib import Path
assert sys.argv[1:3] == ["prefetch", "--config"], sys.argv
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
(Path(os.environ["HF_HOME"]) / "toy.txt").write_text(sys.prefix)
"""


@pytest.fixture
def toy_stage(tmp_path: Path) -> StageSpec:
    project = tmp_path / "stages" / "toy"
    (project / "toystage").mkdir(parents=True)
    (project / "toystage" / "__init__.py").write_text("")
    (project / "toystage" / "__main__.py").write_text(textwrap.dedent(TOY_MAIN))
    (project / "pyproject.toml").write_text(
        '[project]\nname = "toy"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        "dependencies = []\n\n[tool.uv]\npackage = false\n"
    )
    return StageSpec("toy", module="toystage", project=project)


@pytest.fixture
def config(tmp_path: Path) -> Config:
    root = tmp_path / "ssd"
    env = {
        "HOME": str(tmp_path),
        "DUBLADOR_WORK_DIR": str(root / "work"),
        "DUBLADOR_MODELS_DIR": str(root / "models"),
        "DUBLADOR_VENVS_DIR": str(root / "venvs"),
        "UV_CACHE_DIR": str(root / "uv-cache"),
    }
    return load_config(env=env, cwd=tmp_path)


@needs_uv
def test_setup_creates_venv_outside_repo_and_prefetches(
    toy_stage: StageSpec, config: Config
) -> None:
    out = io.StringIO()
    done = run_setup(config, [toy_stage], Console(file=out))
    assert done == ["toy"]
    assert "no config.toml found" in out.getvalue()
    venv = config.paths.venvs_dir / "toy"
    assert (venv / "bin" / "python").exists()
    assert not (toy_stage.project_dir / ".venv").exists()
    assert config.paths.uv_cache_dir.is_dir()
    marker = config.paths.models_dir / "hf" / "toy.txt"
    assert marker.read_text() == str(venv)
    # Idempotent: a second run succeeds and keeps the same venv.
    marker.unlink()
    run_setup(config, [toy_stage], Console(file=out))
    assert marker.is_file()


@needs_uv
def test_skip_prefetch(toy_stage: StageSpec, config: Config) -> None:
    run_setup(config, [toy_stage], Console(file=io.StringIO()), prefetch=False)
    assert (config.paths.venvs_dir / "toy" / "bin" / "python").exists()
    assert not (config.paths.models_dir / "hf").exists()


def test_failing_sync_is_reported(toy_stage: StageSpec, config: Config) -> None:
    (toy_stage.project_dir / "pyproject.toml").write_text("not toml [")
    with pytest.raises(SetupError, match="toy: uv sync failed"):
        run_setup(config, [toy_stage], Console(file=io.StringIO()))


def test_installable_filters(toy_stage: StageSpec) -> None:
    missing = StageSpec("ghost", module="ghost", project=toy_stage.project_dir.parent / "ghost")
    assert installable([toy_stage, missing, PIPELINE[0]]) == [toy_stage]
    assert installable([toy_stage], only=["toy"]) == [toy_stage]
    with pytest.raises(SetupError, match="unknown stage"):
        installable([toy_stage], only=["extract"])


def test_same_filesystem_uses_existing_parent(tmp_path: Path) -> None:
    assert same_filesystem(tmp_path / "a" / "b", tmp_path / "c")
