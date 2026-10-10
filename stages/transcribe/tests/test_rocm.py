"""The HIP-library shim for the ROCm build of CTranslate2."""

from pathlib import Path

import pytest
from transcribe import rocm


def fake_torch_lib(tmp_path: Path, missing: str | None = None) -> Path:
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents=True)
    for name in rocm.SONAMES.values():
        if name != missing:
            (lib / name).write_bytes(b"")
    return lib


def test_make_shim_links_versioned_names(tmp_path: Path) -> None:
    lib = fake_torch_lib(tmp_path)
    shim = tmp_path / "shim"
    assert rocm.make_shim(shim, lib)
    assert (shim / "libamdhip64.so.7").readlink() == lib / "libamdhip64.so"
    assert rocm.make_shim(shim, lib)  # idempotent


def test_make_shim_needs_every_library(tmp_path: Path) -> None:
    lib = fake_torch_lib(tmp_path, missing="libhipblas.so")
    assert not rocm.make_shim(tmp_path / "shim", lib)
    assert not (tmp_path / "shim").exists()


def test_rocm_env_prepends_paths_and_keeps_user_allocator(tmp_path: Path) -> None:
    env = rocm.rocm_env(
        {"LD_LIBRARY_PATH": "/x", "CT2_CUDA_ALLOCATOR": "mine"}, tmp_path / "s", tmp_path / "l"
    )
    assert env["LD_LIBRARY_PATH"] == f"{tmp_path / 's'}:{tmp_path / 'l'}:/x"
    assert env["CT2_CUDA_ALLOCATOR"] == "mine"
    assert env[rocm.MARKER] == "1"
    assert rocm.rocm_env({}, tmp_path, tmp_path)["CT2_CUDA_ALLOCATOR"] == "cub_caching"


def test_reexec_runs_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lib = fake_torch_lib(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(rocm, "torch_lib_dir", lambda: lib)
    monkeypatch.setattr(rocm.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(rocm.sys, "argv", ["/x/__main__.py", "--job-dir", "j"])
    monkeypatch.setattr(rocm.os, "execve", lambda exe, argv, env: calls.append(argv))
    monkeypatch.delenv(rocm.MARKER, raising=False)
    rocm.reexec_with_rocm_libs("transcribe")
    assert calls and calls[0][1:] == ["-m", "transcribe", "--job-dir", "j"]
    monkeypatch.setenv(rocm.MARKER, "1")
    calls.clear()
    rocm.reexec_with_rocm_libs("transcribe")
    assert calls == []
