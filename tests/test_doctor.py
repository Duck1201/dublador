import json
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from dublador import doctor
from dublador.config import Config, load_config
from dublador.doctor import _probe_check, check_disk, check_ollama, check_stages
from dublador.stages import StageSpec


@pytest.fixture
def config(tmp_path: Path) -> Config:
    env = {"HOME": str(tmp_path), "DUBLADOR_WORK_DIR": str(tmp_path / "work")}
    return load_config(env=env, cwd=tmp_path)


def levels(checks: list[doctor.Check]) -> dict[str, str]:
    return {c.name: c.level for c in checks}


def test_ollama_model_present(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "/api/version": {"version": "0.32.5"},
        "/api/tags": {"models": [{"name": "gemma4:e4b-it-qat"}, {"name": "bge-m3:latest"}]},
    }
    monkeypatch.setattr(doctor, "_http_json", lambda url: responses[url.split("11434")[1]])
    assert levels(check_ollama(config)) == {"ollama": "ok", "translator model": "ok"}


def test_ollama_model_missing(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_http_json", lambda url: {"models": [], "version": "x"})
    checks = check_ollama(config)
    assert levels(checks)["translator model"] == "error"
    assert "ollama pull gemma4:e4b-it-qat" in checks[1].detail


def test_ollama_unreachable(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(url: str) -> Any:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(doctor, "_http_json", fail)
    assert levels(check_ollama(config)) == {"ollama": "error"}


def test_disk_thresholds(config: Config) -> None:
    assert levels(check_disk(config, min_free_gb=0, warn_free_gb=0))["disk work_dir"] == "ok"
    assert levels(check_disk(config, min_free_gb=1e9, warn_free_gb=1e9))["disk work_dir"] == "error"
    # All paths on one filesystem: reported once.
    assert sum(c.name.startswith("disk") for c in check_disk(config, 0, 0)) == 1


def test_stage_not_set_up(config: Config) -> None:
    spec = StageSpec("separate", module="separate")
    assert levels(check_stages(config, [spec], None)) == {"stage separate": "warn"}


def test_stage_probe_runs_in_stage_python(config: Config, tmp_path: Path) -> None:
    module = tmp_path / "probestage.py"
    module.write_text(
        "import json\nprint('noise')\n"
        "print(json.dumps({'schema': 'stage-probe/v1', 'stage': 'separate',"
        " 'version': '0.1.0', 'python': '3.12',"
        " 'gpu': {'available': True, 'name': 'RX 7600', 'arch_list': ['gfx1102']}}))\n"
    )
    spec = StageSpec(
        "separate",
        module="probestage",
        python=Path(sys.executable),
        env={"PYTHONPATH": str(tmp_path)},
    )
    checks = check_stages(config, [spec], "gfx1102")
    assert levels(checks) == {"stage separate": "ok"}
    assert "GPU RX 7600" in checks[0].detail


def test_stage_probe_failure(config: Config, tmp_path: Path) -> None:
    (tmp_path / "badstage.py").write_text("raise SystemExit('error: torch not importable')")
    spec = StageSpec(
        "separate",
        module="badstage",
        python=Path(sys.executable),
        env={"PYTHONPATH": str(tmp_path)},
    )
    checks = check_stages(config, [spec], None)
    assert checks[0].level == "error"
    assert "torch not importable" in checks[0].detail


def test_probe_check_levels() -> None:
    base = {"version": "0.1.0", "python": "3.12", "packages": {"torch": "2.11.0+rocm7.2"}}
    assert _probe_check("s", base, "gfx1102").level == "ok"  # stage without GPU
    no_gpu = {**base, "gpu": {"available": False}}
    assert _probe_check("s", no_gpu, "gfx1102").level == "error"
    wrong_arch = {**base, "gpu": {"available": True, "arch_list": ["gfx1100"]}}
    assert _probe_check("s", wrong_arch, "gfx1102").level == "warn"
    ct2_ok = {**base, "gpu": {"available": True, "ctranslate2_devices": 1}}
    assert _probe_check("s", ct2_ok, "gfx1102").level == "ok"
    ct2_cpu = {**base, "gpu": {"available": True, "ctranslate2_devices": 0}}
    check = _probe_check("s", ct2_cpu, "gfx1102")
    assert check.level == "warn" and "CTranslate2 on CPU only" in check.detail


def test_tools_report_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        out = (
            " ... rubberband  A->A  Apply time-stretching\n"
            if "-filters" in cmd
            else f"{cmd[0]} 1.0\n"
        )
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(doctor, "_run", fake_run)
    monkeypatch.setattr(doctor.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    checks = doctor.check_tools()
    assert levels(checks) == {
        "ffmpeg": "ok",
        "ffprobe": "ok",
        "uv": "ok",
        "ffmpeg rubberband": "ok",
    }
    assert json.dumps([c.detail for c in checks]).count("1.0") == 3
