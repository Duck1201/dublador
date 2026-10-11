"""Run the ROCm build of CTranslate2 without a system ROCm install (#19).

The wheel links against versioned HIP libraries (libamdhip64.so.7, ...). The
ROCm torch wheel ships the same libraries under unversioned names, so a
directory of versioned symlinks plus torch/lib on LD_LIBRARY_PATH is enough.
The dynamic loader reads LD_LIBRARY_PATH only at process start, hence the
re-exec.
"""

import importlib.util
import os
import sys
from collections.abc import Mapping
from pathlib import Path

# SONAME the CTranslate2 ROCm wheel needs -> file name in torch/lib.
SONAMES = {
    "libamdhip64.so.7": "libamdhip64.so",
    "libhipblas.so.3": "libhipblas.so",
    "libhiprand.so.1": "libhiprand.so",
}
MARKER = "DUBLADOR_CT2_ROCM_ENV"
# The default allocator gives out-of-memory errors, illegal addresses or
# truncated text on gfx1102 (docs/MODELS.md).
ALLOCATOR = "cub_caching"


def torch_lib_dir() -> Path | None:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    lib = Path(spec.origin).parent / "lib"
    return lib if lib.is_dir() else None


def make_shim(shim: Path, torch_lib: Path) -> bool:
    """Create the versioned symlinks; False when torch lacks a HIP library."""
    if not all((torch_lib / name).is_file() for name in SONAMES.values()):
        return False
    shim.mkdir(parents=True, exist_ok=True)
    for soname, name in SONAMES.items():
        link = shim / soname
        target = torch_lib / name
        if link.is_symlink() and link.readlink() == target:
            continue
        tmp = shim / f".{soname}.tmp"
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(target)
        tmp.replace(link)
    return True


def rocm_env(environ: Mapping[str, str], shim: Path, torch_lib: Path) -> dict[str, str]:
    env = dict(environ)
    paths = [str(shim), str(torch_lib)]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    env.setdefault("CT2_CUDA_ALLOCATOR", ALLOCATOR)
    env[MARKER] = "1"
    return env


def reexec_with_rocm_libs(module: str) -> None:
    """Restart `python -m <module>` once with the HIP libraries visible; no-op
    without them."""
    if os.environ.get(MARKER):
        return
    torch_lib = torch_lib_dir()
    if torch_lib is None or not make_shim(Path(sys.prefix) / "rocm-shim", torch_lib):
        return
    env = rocm_env(os.environ, Path(sys.prefix) / "rocm-shim", torch_lib)
    os.execve(sys.executable, [sys.executable, "-m", module, *sys.argv[1:]], env)
