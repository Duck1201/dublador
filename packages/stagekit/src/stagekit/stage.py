"""Stage entry point: the interface every stage exposes to the orchestrator.

    python -m <stage> --job-dir DIR --config FILE [--only seg_0007,seg_0012]
    python -m <stage> prefetch --config FILE
    python -m <stage> probe

Exit code 0 means success; on failure the error goes to stderr.
"""

import argparse
import json
import re
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from stagekit.progress import Progress
from stagekit.schemas import ContractError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

# Directory of each stage inside the job directory (docs/ARCHITECTURE.md).
STAGE_DIRS: dict[str, str] = {
    "extract": "01_extract",
    "separate": "02_separate",
    "transcribe": "03_transcribe",
    "diarize": "04_diarize",
    "speakers": "05_speakers",
    "translate": "06_translate",
    "synthesize": "07_synthesize",
    "fit": "08_fit",
    "mix": "09_mix",
    "mux": "10_mux",
}

_SEGMENT_ID = re.compile(r"seg_\d{4,}")


class StageError(Exception):
    """Expected failure: reported as a one-line error without a traceback."""


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


@dataclass(frozen=True)
class StageContext:
    """What a stage gets when it runs over a job."""

    name: str
    job_dir: Path
    config: dict[str, Any]
    only: frozenset[str] | None = None
    progress: Progress = field(default_factory=Progress)

    @property
    def section(self) -> dict[str, Any]:
        """This stage's own config section (empty if absent)."""
        return self.config.get(self.name, {})

    @property
    def out_dir(self) -> Path:
        return self.stage_dir(self.name)

    def stage_dir(self, stage: str) -> Path:
        return self.job_dir / STAGE_DIRS[stage]

    def path(self, key: str) -> Path:
        """A directory from the snapshot's [paths] (already absolute)."""
        return Path(self.config["paths"][key])

    def selected(self, segment_id: str) -> bool:
        """Whether a segment is in scope for this run (`--only`)."""
        return self.only is None or segment_id in self.only


RunFn = Callable[[StageContext], None]
PrefetchFn = Callable[[dict[str, Any]], None]
ProbeFn = Callable[[], dict[str, Any]]


@dataclass(frozen=True)
class Invocation:
    command: str  # "run" | "prefetch" | "probe"
    job_dir: Path | None = None
    config: Path | None = None
    only: frozenset[str] | None = None


def _parse_only(value: str) -> frozenset[str]:
    ids = [part.strip() for part in value.split(",") if part.strip()]
    bad = [i for i in ids if not _SEGMENT_ID.fullmatch(i)]
    if bad or not ids:
        raise argparse.ArgumentTypeError(f"invalid segment ids: {value!r}")
    return frozenset(ids)


def parse_args(prog: str, argv: Sequence[str]) -> Invocation:
    if argv and argv[0] == "probe":
        argparse.ArgumentParser(prog=f"{prog} probe").parse_args(argv[1:])
        return Invocation("probe")
    if argv and argv[0] == "prefetch":
        parser = argparse.ArgumentParser(prog=f"{prog} prefetch")
        parser.add_argument("--config", type=Path, required=True)
        ns = parser.parse_args(argv[1:])
        return Invocation("prefetch", config=ns.config)
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", type=_parse_only, default=None)
    ns = parser.parse_args(argv)
    return Invocation("run", job_dir=ns.job_dir, config=ns.config, only=ns.only)


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def main(
    name: str,
    *,
    run: RunFn,
    prefetch: PrefetchFn | None = None,
    probe: ProbeFn | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    """Parse the stage interface, dispatch, and turn errors into exit code 1."""
    if name not in STAGE_DIRS:
        raise ValueError(f"unknown stage: {name}")
    inv = parse_args(name, sys.argv[1:] if argv is None else argv)
    try:
        if inv.command == "probe":
            info = {"schema": "stage-probe/v1", "stage": name, **(probe() if probe else {})}
            print(json.dumps(info, ensure_ascii=False))
            return
        assert inv.config is not None
        config = load_config(inv.config)
        if inv.command == "prefetch":
            if prefetch:
                prefetch(config)
            return
        assert inv.job_dir is not None
        if not inv.job_dir.is_dir():
            raise StageError(f"job directory not found: {inv.job_dir}")
        run(StageContext(name=name, job_dir=inv.job_dir, config=config, only=inv.only))
    except (StageError, ContractError) as e:
        _fail(str(e))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        _fail(f"{type(e).__name__}: {e}")
