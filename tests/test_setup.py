import base64
import hashlib
import io
import shutil
import subprocess
import textwrap
import zipfile
from pathlib import Path

import pytest
from rich.console import Console

from dublador.config import Config, load_config
from dublador.setup import SetupError, fetch_wheel, installable, run_setup, same_filesystem
from dublador.stages import PIPELINE, ExtraWheel, StageSpec

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


def _wheel_bytes() -> bytes:
    """A minimal valid pure-Python wheel: toyextra 0.1."""
    files = {
        "toyextra/__init__.py": b"VALUE = 42\n",
        "toyextra-0.1.dist-info/METADATA": b"Metadata-Version: 2.1\nName: toyextra\nVersion: 0.1\n",
        "toyextra-0.1.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: test\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = []
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(data)}")
    record.append("toyextra-0.1.dist-info/RECORD,,")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        zf.writestr("toyextra-0.1.dist-info/RECORD", "\n".join(record) + "\n")
    return buf.getvalue()


@pytest.fixture
def release_zip(tmp_path: Path) -> ExtraWheel:
    archive = tmp_path / "release" / "wheels.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dist/toyextra-0.1-py3-none-any.whl", _wheel_bytes())
    return ExtraWheel(
        url=archive.as_uri(),
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        member="dist/toyextra-0.1-py3-none-any.whl",
    )


def test_fetch_wheel_downloads_once_and_checks_sha(release_zip: ExtraWheel, tmp_path: Path) -> None:
    downloads = tmp_path / "downloads"
    wheel = fetch_wheel(release_zip, downloads)
    assert wheel.name == "toyextra-0.1-py3-none-any.whl" and wheel.is_file()
    assert fetch_wheel(release_zip, downloads) == wheel  # cached, no download

    bad = ExtraWheel(release_zip.url, "0" * 64, release_zip.member)
    with pytest.raises(SetupError, match="sha256 mismatch"):
        fetch_wheel(bad, tmp_path / "other")
    missing = ExtraWheel(release_zip.url, release_zip.sha256, "dist/nope.whl")
    with pytest.raises(SetupError, match=r"has no dist/nope\.whl"):
        fetch_wheel(missing, tmp_path / "third")


@needs_uv
def test_setup_links_extra_wheels_before_sync(
    toy_stage: StageSpec, release_zip: ExtraWheel, config: Config
) -> None:
    assert toy_stage.project is not None
    (toy_stage.project / "pyproject.toml").write_text(
        '[project]\nname = "toy"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        'dependencies = ["toyextra"]\n\n[tool.uv]\npackage = false\n\n'
        '[tool.uv.sources]\ntoyextra = { path = "wheels/toyextra-0.1-py3-none-any.whl" }\n'
    )
    spec = StageSpec(
        "toy", module="toystage", project=toy_stage.project, extra_wheels=(release_zip,)
    )
    run_setup(config, [spec], Console(file=io.StringIO()), prefetch=False)
    link = toy_stage.project / "wheels" / "toyextra-0.1-py3-none-any.whl"
    assert link.is_symlink() and link.resolve().is_relative_to(config.paths.models_dir)
    python = config.paths.venvs_dir / "toy" / "bin" / "python"
    out = subprocess.run(
        [str(python), "-c", "import toyextra; print(toyextra.VALUE)"],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    assert out.stdout.strip() == "42"
