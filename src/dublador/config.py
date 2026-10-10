"""Configuration loading. See docs/ARCHITECTURE.md#configuração.

Precedence: CLI flags > environment variables > config.toml > defaults.
Sections the orchestrator does not know (e.g. `[synthesize]`) belong to the
stages; they are passed through unvalidated into the job's config snapshot.
"""

import copy
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w
import tomllib

SNAPSHOT_NAME = "config.snapshot.toml"

DEFAULTS: dict[str, dict[str, Any]] = {
    "paths": {
        "work_dir": "~/dublador/work",
        "models_dir": "~/dublador/models",
        "venvs_dir": "~/dublador/venvs",
        "uv_cache_dir": "~/dublador/uv-cache",
    },
    "languages": {"source": "en", "target": "pt-BR"},
    "translate": {"model": "gemma4:e4b-it-qat", "ollama_url": "http://localhost:11434"},
    "fit": {"max_speedup": 1.25, "max_fit_iterations": 2},
    "output": {"format": "mkv"},
    "gpu": {"hsa_override_gfx_version": None},
}

ENV_OVERRIDES: dict[str, str] = {
    "DUBLADOR_WORK_DIR": "paths.work_dir",
    "DUBLADOR_MODELS_DIR": "paths.models_dir",
    "DUBLADOR_VENVS_DIR": "paths.venvs_dir",
    "UV_CACHE_DIR": "paths.uv_cache_dir",
}

OUTPUT_FORMATS = ("mkv", "mp4")

_ENV_VAR = re.compile(r"\$(\w+)|\$\{(\w+)\}")
_GFX_VERSION = re.compile(r"\d+\.\d+\.\d+")


class ConfigError(Exception):
    """Invalid or unreadable configuration."""


@dataclass(frozen=True)
class PathsConfig:
    work_dir: Path
    models_dir: Path
    venvs_dir: Path
    uv_cache_dir: Path


@dataclass(frozen=True)
class LanguagesConfig:
    source: str
    target: str


@dataclass(frozen=True)
class TranslateConfig:
    model: str
    ollama_url: str


@dataclass(frozen=True)
class FitConfig:
    max_speedup: float
    max_fit_iterations: int


@dataclass(frozen=True)
class OutputConfig:
    format: str


@dataclass(frozen=True)
class GpuConfig:
    hsa_override_gfx_version: str | None


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    languages: LanguagesConfig
    translate: TranslateConfig
    fit: FitConfig
    output: OutputConfig
    gpu: GpuConfig
    source_file: Path | None = None
    stage_sections: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_toml_dict(self) -> dict[str, Any]:
        """Effective configuration as TOML-serializable data (None values omitted)."""
        data: dict[str, Any] = {
            "paths": {k: str(v) for k, v in vars(self.paths).items()},
            "languages": dict(vars(self.languages)),
            "translate": dict(vars(self.translate)),
            "fit": dict(vars(self.fit)),
            "output": dict(vars(self.output)),
            "gpu": {k: v for k, v in vars(self.gpu).items() if v is not None},
        }
        for name, section in self.stage_sections.items():
            data[name] = copy.deepcopy(dict(section))
        return data


def find_config_file(env: Mapping[str, str], cwd: Path) -> Path | None:
    """Return the config file to use: $DUBLADOR_CONFIG, ./config.toml, then XDG."""
    explicit = env.get("DUBLADOR_CONFIG")
    if explicit:
        path = _expand_path(explicit, env, base=cwd)
        if not path.is_file():
            raise ConfigError(f"DUBLADOR_CONFIG points to a missing file: {path}")
        return path
    candidates = [cwd / "config.toml"]
    xdg = env.get("XDG_CONFIG_HOME") or str(_home(env) / ".config")
    candidates.append(Path(xdg) / "dublador" / "config.toml")
    return next((c for c in candidates if c.is_file()), None)


def load_config(
    overrides: Mapping[str, object] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> Config:
    """Load and validate the configuration.

    `overrides` maps dotted keys (e.g. "translate.model") to CLI flag values;
    None values mean "flag not given" and are ignored.
    """
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd

    data = copy.deepcopy(DEFAULTS)
    source_file = find_config_file(env, cwd)
    stage_sections: dict[str, dict[str, Any]] = {}
    if source_file is not None:
        for name, section in _read_toml(source_file).items():
            if not isinstance(section, dict):
                raise ConfigError(f"{source_file}: top-level key '{name}' must be a table")
            if name in data:
                data[name].update(section)
            else:
                stage_sections[name] = section

    for var, key in ENV_OVERRIDES.items():
        if env.get(var):
            _set_dotted(data, key, env[var])
    for key, value in (overrides or {}).items():
        if value is not None:
            _set_dotted(data, key, value)

    return _build(data, env, source_file, stage_sections)


def write_snapshot(config: Config, job_dir: Path) -> Path:
    """Write the effective config to <job_dir>/config.snapshot.toml atomically."""
    target = job_dir / SNAPSHOT_NAME
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_bytes(tomli_w.dumps(config.to_toml_dict()).encode())
    os.replace(tmp, target)
    return target


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e
    except OSError as e:
        raise ConfigError(f"{path}: cannot read: {e}") from e


def _set_dotted(data: dict[str, dict[str, Any]], key: str, value: object) -> None:
    section, _, name = key.partition(".")
    if section not in data or not name:
        raise ConfigError(f"unknown config key: {key}")
    data[section][name] = value


def _home(env: Mapping[str, str]) -> Path:
    return Path(env["HOME"]) if env.get("HOME") else Path.home()


def _expand_path(raw: str, env: Mapping[str, str], base: Path | None = None) -> Path:
    """Expand ~ and $VARS. Relative paths resolve against `base`, or are rejected."""

    def var(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        if name not in env:
            raise ConfigError(f"undefined environment variable ${name} in path: {raw}")
        return env[name]

    expanded = _ENV_VAR.sub(var, raw)
    if expanded == "~" or expanded.startswith("~/"):
        expanded = str(_home(env)) + expanded[1:]
    path = Path(expanded)
    if not path.is_absolute() and base is not None:
        path = base / path
    if not path.is_absolute():
        raise ConfigError(f"path must be absolute or start with ~: {raw}")
    return path


def _check_keys(section: str, values: dict[str, Any]) -> None:
    unknown = sorted(set(values) - set(DEFAULTS[section]))
    if unknown:
        raise ConfigError(f"unknown key(s) in [{section}]: {', '.join(unknown)}")


def _str(section: str, key: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section}.{key} must be a non-empty string")
    return value


def _build(
    data: dict[str, dict[str, Any]],
    env: Mapping[str, str],
    source_file: Path | None,
    stage_sections: dict[str, dict[str, Any]],
) -> Config:
    for section, values in data.items():
        _check_keys(section, values)

    paths = data["paths"]
    tr = data["translate"]
    fit = data["fit"]
    fmt = data["output"]["format"]
    gfx = data["gpu"]["hsa_override_gfx_version"]

    translate = TranslateConfig(**{k: _str("translate", k, v) for k, v in tr.items()})
    if not translate.ollama_url.startswith(("http://", "https://")):
        raise ConfigError("translate.ollama_url must start with http:// or https://")

    speedup = fit["max_speedup"]
    if isinstance(speedup, bool) or not isinstance(speedup, int | float) or speedup <= 1:
        raise ConfigError("fit.max_speedup must be a number greater than 1")
    iterations = fit["max_fit_iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ConfigError("fit.max_fit_iterations must be a non-negative integer")

    if fmt not in OUTPUT_FORMATS:
        raise ConfigError(f"output.format must be one of: {', '.join(OUTPUT_FORMATS)}")
    if gfx is not None and not (isinstance(gfx, str) and _GFX_VERSION.fullmatch(gfx)):
        raise ConfigError('gpu.hsa_override_gfx_version must look like "11.0.0"')

    return Config(
        paths=PathsConfig(**{k: _expand_path(_str("paths", k, v), env) for k, v in paths.items()}),
        languages=LanguagesConfig(
            **{k: _str("languages", k, v) for k, v in data["languages"].items()}
        ),
        translate=translate,
        fit=FitConfig(max_speedup=float(speedup), max_fit_iterations=iterations),
        output=OutputConfig(format=fmt),
        gpu=GpuConfig(hsa_override_gfx_version=gfx),
        source_file=source_file,
        stage_sections=stage_sections,
    )
