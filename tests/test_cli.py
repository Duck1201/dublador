from pathlib import Path

import pytest
from typer.testing import CliRunner

from dublador import __version__
from dublador.cli import app

runner = CliRunner()

COMMANDS = ["run", "resume", "status", "clean", "setup", "doctor"]
STUBS = ["clean"]


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("command", STUBS)
def test_stub_commands_fail_until_implemented(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_run_reports_config_errors(tmp_path: Path) -> None:
    config = tmp_path / "bad.toml"
    config.write_text('[output]\nformat = "avi"\n')
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    result = runner.invoke(app, ["run", str(video)], env={"DUBLADOR_CONFIG": str(config)})
    assert result.exit_code == 2
    assert "output.format" in result.output


def test_run_requires_video() -> None:
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2
    assert "missing VIDEO" in result.output
