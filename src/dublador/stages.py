"""Stage registry: what each stage reads and writes, and how it is run."""

import hashlib
import inspect
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dublador import __version__, ffmpeg
from dublador.config import Config
from dublador.manifest import Manifest

# stages/<name>/ projects live next to src/ in the repository.
STAGES_DIR = Path(__file__).resolve().parents[2] / "stages"

InProcessFn = Callable[[Path, Manifest, Config], None]


@dataclass(frozen=True)
class ExtraWheel:
    """A wheel from a release archive that no package index serves (e.g. the ROCm
    build of CTranslate2). `dublador setup` downloads it into models_dir and links
    it as stages/<name>/wheels/<file> before `uv sync`; the stage's pyproject points
    a path source there, so the lockfile holds a relative path and the wheel hash,
    and `uv sync`/`uv run` keep installing this build."""

    url: str
    sha256: str  # of the archive at `url`
    member: str  # path of the wheel inside the zip archive


@dataclass(frozen=True)
class StageSpec:
    """How to run one stage and what its result depends on.

    A stage with `module` runs as `python -m <module>` in its own venv; one
    with `function` runs inside the orchestrator (ffmpeg-only stages).
    `inputs`/`outputs` are paths relative to the job directory. The inputs
    hash covers the input files, the stage's config sections, the listed
    run options, the video (if `uses_video`) and the stage version.
    """

    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    config_sections: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    uses_video: bool = False
    writes_output_video: bool = False
    module: str | None = None
    function: InProcessFn | None = None
    project: Path | None = None  # default: stages/<name>
    python: Path | None = None  # default: <venvs_dir>/<name>/bin/python
    env: Mapping[str, str] = field(default_factory=dict)
    extra_wheels: tuple[ExtraWheel, ...] = ()
    version: str | None = None  # default: derived from the stage's code (see resolved_version)

    @property
    def project_dir(self) -> Path:
        return self.project or STAGES_DIR / self.name

    def python_path(self, config: Config) -> Path:
        return self.python or config.paths.venvs_dir / self.name / "bin" / "python"

    def resolved_version(self) -> str | None:
        """Declared version plus a digest of the code, so editing a stage without
        bumping its version still invalidates cached results."""
        if self.version is not None:
            return self.version
        if self.function is not None:
            source = inspect.getsourcefile(self.function)
            files = [Path(source)] if source else []
            return f"{__version__}+src.{_digest(files, Path(source or '.').parent)}"
        pyproject = self.project_dir / "pyproject.toml"
        if not pyproject.is_file():
            return None
        with pyproject.open("rb") as f:
            declared = tomllib.load(f).get("project", {}).get("version", "0")
        src = self.project_dir / "src"
        files = [pyproject, self.project_dir / "uv.lock"]
        files += [p for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        version = f"{declared}+src.{_digest(files, self.project_dir)}"
        if self.extra_wheels:
            wheels = "".join(w.sha256 + w.member for w in self.extra_wheels).encode()
            version += f".wheels.{hashlib.sha256(wheels).hexdigest()[:8]}"
        return version


def _digest(files: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(f for f in files if f.is_file()):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


_SEP = "02_separate"
_CT2_ROCM = ExtraWheel(
    url="https://github.com/OpenNMT/CTranslate2/releases/download/v4.8.2/rocm-python-wheels-Linux.zip",
    sha256="b469e765f74ef85fb97bf4fe2347c8c6296dda00d7f5cf20658f23e95004a925",
    member="temp-linux/ctranslate2-4.8.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
)

# M1 pipeline. diarize and fit join in M2/M3.
PIPELINE: tuple[StageSpec, ...] = (
    StageSpec(
        "extract",
        outputs=(ffmpeg.AUDIO_WAV, ffmpeg.PROBE_JSON),
        options=("audio_track",),
        uses_video=True,
        function=ffmpeg.extract,
    ),
    StageSpec(
        "separate",
        inputs=(ffmpeg.AUDIO_WAV,),
        outputs=(f"{_SEP}/vocals.wav", f"{_SEP}/accompaniment.wav"),
        module="separate",
    ),
    StageSpec(
        "transcribe",
        inputs=(f"{_SEP}/vocals.wav",),
        outputs=("03_transcribe/transcript.json",),
        config_sections=("languages",),
        module="transcribe",
        # Whisper on the GPU: PyPI only has CPU/CUDA builds of CTranslate2 (#19).
        extra_wheels=(_CT2_ROCM,),
    ),
    StageSpec(
        "speakers",
        inputs=(f"{_SEP}/vocals.wav", f"{_SEP}/accompaniment.wav", "03_transcribe/transcript.json"),
        outputs=("05_speakers/speakers.json",),
        module="speakers",
    ),
    StageSpec(
        "translate",
        inputs=("03_transcribe/transcript.json", "05_speakers/speakers.json"),
        outputs=("06_translate/translation.json",),
        config_sections=("translate", "languages"),
        module="translate",
    ),
    StageSpec(
        "synthesize",
        inputs=("06_translate/translation.json", "05_speakers/speakers.json"),
        outputs=("07_synthesize/synthesis.json",),
        module="synthesize",
    ),
    StageSpec(
        "mix",
        inputs=(
            "07_synthesize/synthesis.json",
            "06_translate/translation.json",
            f"{_SEP}/accompaniment.wav",
        ),
        outputs=(ffmpeg.DUB_WAV,),
        module="mix",
    ),
    StageSpec(
        "mux",
        inputs=(ffmpeg.DUB_WAV, ffmpeg.PROBE_JSON),
        config_sections=("output", "languages"),
        options=("output_path", "audio_track"),
        uses_video=True,
        writes_output_video=True,
        function=ffmpeg.mux,
    ),
)
