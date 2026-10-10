"""End-to-end orchestration with a synthetic video and fake stages."""

import io
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from dublador import stages
from dublador.cli import app
from dublador.config import Config, load_config, write_snapshot
from dublador.manifest import Manifest
from dublador.runner import PipelineError, inputs_hash, open_job, run_pipeline
from dublador.stages import StageSpec

Pipeline = Callable[..., tuple[StageSpec, ...]]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return load_config(
        env={"HOME": str(tmp_path), "DUBLADOR_WORK_DIR": str(tmp_path / "work")}, cwd=tmp_path
    )


def run(
    video: Path, config: Config, specs: tuple[StageSpec, ...], output: Path | None = None
) -> tuple[Manifest, str]:
    job, manifest = open_job(video, config, specs, output=output, audio_track=None)
    write_snapshot(config, job.dir)
    buffer = io.StringIO()
    try:
        run_pipeline(job, manifest, config, specs, Console(file=buffer, width=200))
    finally:
        manifest = Manifest.load(job.manifest_path)
    return manifest, buffer.getvalue()


def audio_streams(path: Path) -> list[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return [s for s in json.loads(out)["streams"] if s["codec_type"] == "audio"]


def statuses(manifest: Manifest) -> dict[str, str]:
    return {name: entry["status"] for name, entry in manifest.data["stages"].items()}


def test_full_run_produces_dubbed_mkv(
    video: Path, config: Config, fake_pipeline: Pipeline, tmp_path: Path
) -> None:
    output = tmp_path / "out" / "dubbed.mkv"
    output.parent.mkdir()
    manifest, log = run(video, config, fake_pipeline(), output=output)
    assert set(statuses(manifest).values()) == {"done"}
    assert "mixing" in log
    tracks = audio_streams(output)
    assert [t["tags"]["language"] for t in tracks] == ["por", "eng"]
    assert tracks[0]["disposition"]["default"] == 1
    assert tracks[1]["disposition"]["default"] == 0
    assert list(output.parent.iterdir()) == [output]


def test_default_output_next_to_video(video: Path, config: Config, fake_pipeline: Pipeline) -> None:
    manifest, _ = run(video, config, fake_pipeline())
    assert manifest.output_path == video.with_name("Sample Talk.ptbr.mkv")
    assert manifest.output_path.is_file()
    manifest.output_path.unlink()


def test_failure_then_resume_skips_finished_stages(
    video: Path, config: Config, fake_pipeline: Pipeline, tmp_path: Path
) -> None:
    output = tmp_path / "o.mkv"
    with pytest.raises(PipelineError, match="forced failure in mix"):
        run(video, config, fake_pipeline(fail="mix"), output=output)
    job_dir = next((tmp_path / "work").iterdir())
    manifest = Manifest.load(job_dir / "manifest.json")
    assert statuses(manifest) == {
        "extract": "done",
        "separate": "done",
        "mix": "failed",
        "mux": "pending",
    }
    assert manifest.data["stages"]["mix"]["error"] == "mix failed (exit 1): forced failure in mix"
    assert (
        "library noise" not in (job_dir / "logs" / "mix.log").read_text()
    )  # failed before printing

    manifest, log = run(video, config, fake_pipeline(), output=output)
    assert "extract (up to date)" in log
    assert "separate (up to date)" in log
    assert set(statuses(manifest).values()) == {"done"}
    assert "library noise on stdout" in (job_dir / "logs" / "mix.log").read_text()


def test_changed_input_or_missing_output_reruns(
    video: Path, config: Config, fake_pipeline: Pipeline, tmp_path: Path
) -> None:
    output = tmp_path / "o.mkv"
    specs = fake_pipeline()
    run(video, config, specs, output=output)
    job_dir = next((tmp_path / "work").iterdir())

    (job_dir / "09_mix" / "dub.wav").unlink()
    _, log = run(video, config, specs, output=output)
    assert "separate (up to date)" in log
    assert "mix (up to date)" not in log

    output.unlink()
    _, log = run(video, config, specs, output=output)
    assert "mix (up to date)" in log
    assert output.is_file()


def test_inputs_hash_tracks_config_section(
    video: Path, config: Config, fake_pipeline: Pipeline
) -> None:
    specs = fake_pipeline()
    job, manifest = open_job(video, config, specs, output=None, audio_track=None)
    mux = specs[-1]
    before = inputs_hash(mux, job, manifest, config)
    other = load_config(
        {"output.format": "mp4"},
        env={"HOME": "/h", "DUBLADOR_WORK_DIR": str(config.paths.work_dir)},
    )
    assert inputs_hash(mux, job, manifest, other) != before
    assert inputs_hash(specs[1], job, manifest, other) == inputs_hash(
        specs[1], job, manifest, config
    )


def test_missing_stage_venv(video: Path, config: Config) -> None:
    specs = (stages.PIPELINE[0], stages.PIPELINE[1])
    with pytest.raises(PipelineError, match="dublador setup"):
        run(video, config, specs)


def test_cli_run_status_resume(
    video: Path, fake_pipeline: Pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env = {
        "DUBLADOR_WORK_DIR": str(tmp_path / "work"),
        "DUBLADOR_CONFIG": "",
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
    }
    runner = CliRunner()
    output = tmp_path / "cli.mkv"

    monkeypatch.setattr(stages, "PIPELINE", fake_pipeline(fail="separate"))
    result = runner.invoke(app, ["run", str(video), "--output", str(output)], env=env)
    assert result.exit_code == 1, result.output
    assert "forced failure in separate" in result.output

    result = runner.invoke(app, ["status"], env=env)
    assert result.exit_code == 0
    assert "sample-talk-" in result.output and "failed" in result.output

    monkeypatch.setattr(stages, "PIPELINE", fake_pipeline())
    result = runner.invoke(app, ["resume"], env=env)
    assert result.exit_code == 0, result.output
    assert "extract (up to date)" in result.output
    assert output.is_file()

    job_id = next((tmp_path / "work").iterdir()).name
    result = runner.invoke(app, ["status", job_id], env=env)
    assert result.exit_code == 0
    assert result.output.count("done") >= 4


def test_stage_version_tracks_source_code(tmp_path: Path) -> None:
    project = tmp_path / "toy"
    (project / "src" / "toy").mkdir(parents=True)
    (project / "pyproject.toml").write_text('[project]\nname = "toy"\nversion = "0.1.0"\n')
    module = project / "src" / "toy" / "__main__.py"
    module.write_text("print(1)\n")
    spec = StageSpec("toy", module="toy", project=project)
    before = spec.resolved_version()
    assert before is not None and before.startswith("0.1.0+src.")
    (project / "src" / "toy" / "__pycache__").mkdir()
    (project / "src" / "toy" / "__pycache__" / "x.pyc").write_bytes(b"junk")
    assert spec.resolved_version() == before
    module.write_text("print(2)\n")
    assert spec.resolved_version() != before


def test_in_process_stage_version_tracks_module() -> None:
    version = stages.PIPELINE[0].resolved_version()
    assert version is not None and "+src." in version


def test_stage_env_defaults(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    from dublador.runner import stage_env

    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    env = stage_env(config)
    assert env["MIOPEN_FIND_MODE"] == "FAST"
    assert env["HF_HOME"] == str(config.paths.models_dir / "hf")
    assert "HSA_OVERRIDE_GFX_VERSION" not in env  # only when set in the config
    monkeypatch.setenv("MIOPEN_FIND_MODE", "NORMAL")
    assert stage_env(config)["MIOPEN_FIND_MODE"] == "NORMAL"
