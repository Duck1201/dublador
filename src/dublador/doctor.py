"""`dublador doctor`: checks the machine before a long job."""

import grp
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dublador.config import Config
from dublador.runner import stage_env
from dublador.setup import same_filesystem
from dublador.stages import StageSpec

Level = Literal["ok", "warn", "error"]

GB = 1024**3
PROBE_TIMEOUT_S = 180


@dataclass(frozen=True)
class Check:
    name: str
    level: Level
    detail: str


def _run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kwargs)


def _http_json(url: str, timeout: float = 3.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def check_tools() -> list[Check]:
    checks = []
    for tool in ("ffmpeg", "ffprobe", "uv"):
        if shutil.which(tool) is None:
            checks.append(Check(tool, "error", "not found in PATH"))
            continue
        flag = "--version" if tool == "uv" else "-version"
        checks.append(Check(tool, "ok", _first_line(_run([tool, flag]).stdout)))
    if shutil.which("ffmpeg"):
        filters = _run(["ffmpeg", "-hide_banner", "-filters"]).stdout
        if re.search(r"\srubberband\s", filters):
            checks.append(Check("ffmpeg rubberband", "ok", "filter available (time-stretch)"))
        else:
            checks.append(
                Check("ffmpeg rubberband", "warn", "filter missing; time-fit will use atempo")
            )
    return checks


def check_disk(config: Config, min_free_gb: float, warn_free_gb: float) -> list[Check]:
    paths = config.paths
    checks = []
    if same_filesystem(paths.uv_cache_dir, paths.venvs_dir):
        checks.append(Check("uv cache", "ok", "same filesystem as venvs_dir"))
    else:
        checks.append(
            Check("uv cache", "warn", "uv_cache_dir and venvs_dir on different filesystems")
        )

    seen: set[int] = set()
    for key in ("work_dir", "models_dir", "venvs_dir", "uv_cache_dir"):
        path: Path = getattr(paths, key)
        existing = path
        while not existing.exists():
            existing = existing.parent
        device = existing.stat().st_dev
        if device in seen:
            continue
        seen.add(device)
        free = shutil.disk_usage(existing).free / GB
        level: Level = "error" if free < min_free_gb else "warn" if free < warn_free_gb else "ok"
        checks.append(Check(f"disk {key}", level, f"{free:.0f} GB free at {path}"))
    return checks


def _model_names(tags: Any) -> set[str]:
    names = set()
    for model in tags.get("models", []):
        name = model.get("name", "")
        names.add(name)
        if name.endswith(":latest"):
            names.add(name.removesuffix(":latest"))
    return names


def check_ollama(config: Config) -> list[Check]:
    url = config.translate.ollama_url.rstrip("/")
    try:
        version = _http_json(f"{url}/api/version").get("version", "?")
        tags = _http_json(f"{url}/api/tags")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return [Check("ollama", "error", f"not reachable at {url}: {e}")]
    checks = [Check("ollama", "ok", f"{version} at {url}")]
    model = config.translate.model
    if model in _model_names(tags):
        checks.append(Check("translator model", "ok", model))
    else:
        checks.append(
            Check("translator model", "error", f"{model} not pulled; run `ollama pull {model}`")
        )
    return checks


def gpu_arch() -> str | None:
    """gfx name of the first GPU agent reported by rocminfo, if available."""
    if shutil.which("rocminfo") is None:
        return None
    match = re.search(r"Name:\s+(gfx\w+)", _run(["rocminfo"]).stdout)
    return match.group(1) if match else None


def check_gpu() -> list[Check]:
    checks = []
    if Path("/dev/kfd").exists():
        checks.append(Check("/dev/kfd", "ok", "present"))
    else:
        checks.append(Check("/dev/kfd", "error", "missing; amdgpu/KFD driver not loaded"))
    try:
        render_gid = grp.getgrnam("render").gr_gid
    except KeyError:
        render_gid = None
    if render_gid is not None and render_gid in os.getgroups():
        checks.append(Check("render group", "ok", "user is a member"))
    else:
        checks.append(Check("render group", "error", "user not in group `render`"))
    arch = gpu_arch()
    if arch:
        checks.append(Check("gpu", "ok", arch))
    else:
        checks.append(Check("gpu", "warn", "rocminfo unavailable or no GPU agent found"))
    return checks


def check_stages(config: Config, specs: Sequence[StageSpec], arch: str | None) -> list[Check]:
    checks = []
    for spec in specs:
        if not spec.module:
            continue
        name = f"stage {spec.name}"
        python = spec.python_path(config)
        if not python.exists():
            checks.append(Check(name, "warn", "not set up; run `dublador setup`"))
            continue
        try:
            result = _run(
                [str(python), "-m", spec.module, "probe"],
                env=stage_env(config, dict(spec.env)),
                timeout=PROBE_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            checks.append(Check(name, "error", f"probe timed out after {PROBE_TIMEOUT_S} s"))
            continue
        if result.returncode != 0:
            lines = result.stderr.strip().splitlines()
            checks.append(Check(name, "error", f"probe failed: {lines[-1] if lines else '?'}"))
            continue
        try:
            info = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            checks.append(Check(name, "error", "probe printed no JSON"))
            continue
        checks.append(_probe_check(name, info, arch))
    return checks


def _probe_check(name: str, info: dict[str, Any], arch: str | None) -> Check:
    parts = [f"v{info.get('version', '?')}", f"python {info.get('python', '?')}"]
    parts += [f"{pkg} {ver}" for pkg, ver in info.get("packages", {}).items()]
    gpu = info.get("gpu")
    if gpu is None:
        return Check(name, "ok", ", ".join(parts))
    if not gpu.get("available"):
        return Check(name, "error", ", ".join([*parts, "GPU not available"]))
    arch_list = gpu.get("arch_list", [])
    if arch and arch_list and arch not in arch_list:
        return Check(name, "warn", ", ".join([*parts, f"{arch} not in wheel arch list"]))
    return Check(name, "ok", ", ".join([*parts, f"GPU {gpu.get('name') or 'available'}"]))


def run_checks(
    config: Config,
    specs: Sequence[StageSpec],
    *,
    min_free_gb: float = 10,
    warn_free_gb: float = 50,
) -> list[Check]:
    source = str(config.source_file) if config.source_file else "none (defaults)"
    checks = [Check("config", "ok" if config.source_file else "warn", source)]
    checks += check_tools()
    checks += check_disk(config, min_free_gb, warn_free_gb)
    checks += check_ollama(config)
    gpu = check_gpu()
    checks += gpu
    arch = next((c.detail for c in gpu if c.name == "gpu" and c.level == "ok"), None)
    checks += check_stages(config, specs, arch)
    return checks
