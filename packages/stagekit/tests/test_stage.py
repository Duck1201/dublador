import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from stagekit import StageContext, StageError, main
from stagekit.stage import parse_args


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "config.snapshot.toml"
    path.write_text('[paths]\nmodels_dir = "/models"\n[separate]\nchunk_duration = 300\n')
    return path


@pytest.fixture
def job_dir(tmp_path: Path) -> Path:
    path = tmp_path / "job"
    path.mkdir()
    return path


def test_parse_run_args() -> None:
    inv = parse_args(
        "separate", ["--job-dir", "/j", "--config", "/c", "--only", "seg_0007,seg_0012"]
    )
    assert inv.command == "run"
    assert inv.job_dir == Path("/j")
    assert inv.only == frozenset({"seg_0007", "seg_0012"})


def test_parse_prefetch_and_probe() -> None:
    assert parse_args("separate", ["prefetch", "--config", "/c"]).config == Path("/c")
    assert parse_args("separate", ["probe"]).command == "probe"


@pytest.mark.parametrize("only", ["seg_7", "", "seg_0001,,x"])
def test_invalid_only_is_rejected(only: str) -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args("separate", ["--job-dir", "/j", "--config", "/c", "--only", only])
    assert exc.value.code == 2


def test_run_receives_context(config: Path, job_dir: Path) -> None:
    seen: list[StageContext] = []
    main("separate", run=seen.append, argv=["--job-dir", str(job_dir), "--config", str(config)])
    ctx = seen[0]
    assert ctx.section == {"chunk_duration": 300}
    assert ctx.out_dir == job_dir / "02_separate"
    assert ctx.stage_dir("extract") == job_dir / "01_extract"
    assert ctx.path("models_dir") == Path("/models")
    assert ctx.selected("seg_0001")


def test_only_restricts_selection(config: Path, job_dir: Path) -> None:
    seen: list[StageContext] = []
    argv = ["--job-dir", str(job_dir), "--config", str(config), "--only", "seg_0002"]
    main("separate", run=seen.append, argv=argv)
    assert seen[0].selected("seg_0002")
    assert not seen[0].selected("seg_0001")


def test_stage_error_exits_1_without_traceback(
    config: Path, job_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def run(ctx: StageContext) -> None:
        raise StageError("vocals.wav is empty")

    with pytest.raises(SystemExit) as exc:
        main("separate", run=run, argv=["--job-dir", str(job_dir), "--config", str(config)])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err == "error: vocals.wav is empty\n"


def test_unexpected_error_exits_1_with_traceback(
    config: Path, job_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def run(ctx: StageContext) -> None:
        raise KeyError("boom")

    with pytest.raises(SystemExit) as exc:
        main("separate", run=run, argv=["--job-dir", str(job_dir), "--config", str(config)])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "Traceback" in err
    assert err.endswith("error: KeyError: 'boom'\n")


def test_missing_job_dir(config: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(
            "separate",
            run=lambda ctx: None,
            argv=["--job-dir", str(tmp_path / "x"), "--config", str(config)],
        )


def test_prefetch_gets_config(config: Path) -> None:
    got: list[dict[str, Any]] = []
    main(
        "separate",
        run=lambda ctx: None,
        prefetch=got.append,
        argv=["prefetch", "--config", str(config)],
    )
    assert got[0]["separate"] == {"chunk_duration": 300}


def test_probe_prints_json(capsys: pytest.CaptureFixture[str]) -> None:
    main("separate", run=lambda ctx: None, probe=lambda: {"version": "0.1.0"}, argv=["probe"])
    info = json.loads(capsys.readouterr().out)
    assert info == {"schema": "stage-probe/v1", "stage": "separate", "version": "0.1.0"}


def test_unknown_stage_name() -> None:
    with pytest.raises(ValueError, match="unknown stage"):
        main("nope", run=lambda ctx: None, argv=["probe"])


def test_subprocess_contract(tmp_path: Path, config: Path, job_dir: Path) -> None:
    """A stage run as `python -m` emits JSON-line progress and writes atomically."""
    pkg = tmp_path / "src" / "toystage"
    pkg.mkdir(parents=True)
    (pkg / "__main__.py").write_text(
        textwrap.dedent(
            """
            from stagekit import StageContext, atomic_write_text, main

            def run(ctx: StageContext) -> None:
                for i in range(3):
                    ctx.progress.progress(i + 1, 3)
                atomic_write_text(ctx.out_dir / "done.txt", "ok")

            main("separate", run=run)
            """
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "toystage", "--job-dir", str(job_dir), "--config", str(config)],
        cwd=tmp_path / "src",
        capture_output=True,
        text=True,
        check=True,
    )
    events = [json.loads(line) for line in result.stdout.splitlines()]
    assert events[-1] == {"event": "progress", "done": 3, "total": 3}
    assert (job_dir / "02_separate" / "done.txt").read_text() == "ok"
