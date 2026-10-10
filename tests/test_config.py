import tomllib
from pathlib import Path

import pytest

from dublador.config import ConfigError, find_config_file, load_config, write_snapshot


@pytest.fixture
def home(tmp_path: Path) -> Path:
    path = tmp_path / "home"
    path.mkdir()
    return path


@pytest.fixture
def cwd(tmp_path: Path) -> Path:
    path = tmp_path / "cwd"
    path.mkdir()
    return path


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_defaults_without_config_file(home: Path, cwd: Path) -> None:
    config = load_config(env={"HOME": str(home)}, cwd=cwd)
    assert config.source_file is None
    assert config.paths.work_dir == home / "dublador" / "work"
    assert config.languages.target == "pt-BR"
    assert config.fit.max_speedup == 1.25
    assert config.gpu.hsa_override_gfx_version is None


def test_file_overrides_defaults(home: Path, cwd: Path) -> None:
    write(cwd / "config.toml", '[translate]\nmodel = "qwen3:x"\n[fit]\nmax_speedup = 2\n')
    config = load_config(env={"HOME": str(home)}, cwd=cwd)
    assert config.translate.model == "qwen3:x"
    assert config.translate.ollama_url == "http://localhost:11434"
    assert config.fit.max_speedup == 2.0


def test_env_overrides_file(home: Path, cwd: Path) -> None:
    write(cwd / "config.toml", '[paths]\nwork_dir = "/from/file"\n')
    env = {"HOME": str(home), "DUBLADOR_WORK_DIR": "/from/env", "UV_CACHE_DIR": "/uv"}
    config = load_config(env=env, cwd=cwd)
    assert config.paths.work_dir == Path("/from/env")
    assert config.paths.uv_cache_dir == Path("/uv")


def test_flags_override_env_and_file(home: Path, cwd: Path) -> None:
    write(cwd / "config.toml", '[paths]\nwork_dir = "/from/file"\n[output]\nformat = "mp4"\n')
    env = {"HOME": str(home), "DUBLADOR_WORK_DIR": "/from/env"}
    overrides = {"paths.work_dir": "/from/flag", "output.format": "mkv", "translate.model": None}
    config = load_config(overrides, env=env, cwd=cwd)
    assert config.paths.work_dir == Path("/from/flag")
    assert config.output.format == "mkv"
    assert config.translate.model == "gemma4:e4b-it-qat"


def test_config_file_lookup_order(home: Path, cwd: Path, tmp_path: Path) -> None:
    xdg = write(home / ".config" / "dublador" / "config.toml", "")
    assert find_config_file({"HOME": str(home)}, cwd) == xdg
    local = write(cwd / "config.toml", "")
    assert find_config_file({"HOME": str(home)}, cwd) == local
    explicit = write(tmp_path / "other.toml", "")
    env = {"HOME": str(home), "DUBLADOR_CONFIG": str(explicit)}
    assert find_config_file(env, cwd) == explicit


def test_relative_explicit_config_resolves_against_cwd(home: Path, cwd: Path) -> None:
    config_file = write(cwd / "conf" / "mine.toml", "")
    env = {"HOME": str(home), "DUBLADOR_CONFIG": "conf/mine.toml"}
    assert find_config_file(env, cwd) == config_file


def test_missing_explicit_config_file(home: Path, cwd: Path) -> None:
    with pytest.raises(ConfigError, match="missing file"):
        load_config(env={"HOME": str(home), "DUBLADOR_CONFIG": "/nope.toml"}, cwd=cwd)


def test_path_expansion(home: Path, cwd: Path) -> None:
    write(cwd / "config.toml", '[paths]\nmodels_dir = "$DATA/models"\nvenvs_dir = "~/v"\n')
    config = load_config(env={"HOME": str(home), "DATA": "/data"}, cwd=cwd)
    assert config.paths.models_dir == Path("/data/models")
    assert config.paths.venvs_dir == home / "v"


@pytest.mark.parametrize(
    ("toml", "message"),
    [
        ('[paths]\nwork_dir = "relative/dir"\n', "must be absolute"),
        ('[paths]\nwork_dir = "$MISSING/x"\n', "undefined environment variable"),
        ('[output]\nformat = "avi"\n', "output.format"),
        ("[fit]\nmax_speedup = 0.9\n", "fit.max_speedup"),
        ("[fit]\nmax_speedup = true\n", "fit.max_speedup"),
        ("[fit]\nmax_fit_iterations = -1\n", "fit.max_fit_iterations"),
        ('[translate]\nollama_url = "localhost"\n', "ollama_url"),
        ("[translate]\nmodel = 3\n", "translate.model"),
        ('[gpu]\nhsa_override_gfx_version = "gfx1100"\n', "hsa_override_gfx_version"),
        ("[fit]\nmax_speed = 1.5\n", "unknown key"),
        ('fit = "x"\n', "must be a table"),
        ("[fit\n", "invalid TOML"),
    ],
)
def test_validation_errors(home: Path, cwd: Path, toml: str, message: str) -> None:
    write(cwd / "config.toml", toml)
    with pytest.raises(ConfigError, match=message):
        load_config(env={"HOME": str(home)}, cwd=cwd)


def test_unknown_override_key(home: Path, cwd: Path) -> None:
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config({"nope.key": 1}, env={"HOME": str(home)}, cwd=cwd)


def test_snapshot_round_trip_keeps_stage_sections(home: Path, cwd: Path, tmp_path: Path) -> None:
    write(
        cwd / "config.toml",
        '[gpu]\nhsa_override_gfx_version = "11.0.0"\n[synthesize]\nbackend = "qwen3_tts"\n',
    )
    config = load_config(env={"HOME": str(home)}, cwd=cwd)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    snapshot = write_snapshot(config, job_dir)
    assert snapshot.name == "config.snapshot.toml"
    assert list(job_dir.iterdir()) == [snapshot]
    data = tomllib.loads(snapshot.read_text())
    assert data["synthesize"] == {"backend": "qwen3_tts"}
    assert data["gpu"] == {"hsa_override_gfx_version": "11.0.0"}
    assert data["paths"]["work_dir"] == str(home / "dublador" / "work")
    # The snapshot is itself a valid config file that reproduces the same config.
    reloaded = load_config(env={"HOME": str(home), "DUBLADOR_CONFIG": str(snapshot)}, cwd=cwd)
    assert reloaded.to_toml_dict() == config.to_toml_dict()


def test_snapshot_omits_unset_gpu_override(home: Path, cwd: Path, tmp_path: Path) -> None:
    config = load_config(env={"HOME": str(home)}, cwd=cwd)
    data = tomllib.loads(write_snapshot(config, tmp_path).read_text())
    assert data["gpu"] == {}


def test_example_config_is_valid(home: Path, cwd: Path) -> None:
    example = Path(__file__).parents[1] / "config.example.toml"
    config = load_config(env={"HOME": str(home), "DUBLADOR_CONFIG": str(example)}, cwd=cwd)
    assert config.source_file == example


def test_shared_section_passes_stage_keys_through(home: Path, cwd: Path, tmp_path: Path) -> None:
    write(cwd / "config.toml", '[translate]\nmodel = "m"\nbatch_size = 4\n')
    config = load_config({"translate.model": "flag"}, env={"HOME": str(home)}, cwd=cwd)
    assert config.translate.model == "flag"
    data = tomllib.loads(write_snapshot(config, tmp_path).read_text())
    assert data["translate"] == {
        "model": "flag",
        "ollama_url": "http://localhost:11434",
        "batch_size": 4,
    }
