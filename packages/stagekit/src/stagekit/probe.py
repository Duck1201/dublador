"""Environment report for `python -m <stage> probe` (shown by `dublador doctor`)."""

import platform
from importlib import metadata
from typing import Any


def package_versions(*names: str) -> dict[str, str]:
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def torch_gpu() -> dict[str, Any]:
    """GPU as seen by torch (ROCm shows up as `cuda`). Imports torch lazily."""
    import torch  # pyright: ignore[reportMissingImports]  # only in stage venvs

    if not torch.cuda.is_available():
        return {"available": False, "name": None, "arch_list": torch.cuda.get_arch_list()}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "arch": getattr(torch.cuda.get_device_properties(0), "gcnArchName", None),
        "arch_list": torch.cuda.get_arch_list(),
    }


def report(version: str, packages: tuple[str, ...] = (), *, gpu: bool = True) -> dict[str, Any]:
    info: dict[str, Any] = {
        "version": version,
        "python": platform.python_version(),
        "packages": package_versions(*packages),
    }
    if gpu:
        info["gpu"] = torch_gpu()
    return info
