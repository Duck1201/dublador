"""Stage registry: what each stage reads and writes, and how it is run."""

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dublador import __version__, ffmpeg
from dublador.config import Config
from dublador.manifest import Manifest

# stages/<name>/ projects live next to src/ in the repository.
STAGES_DIR = Path(__file__).resolve().parents[2] / "stages"

InProcessFn = Callable[[Path, Manifest, Config], None]


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
    python: Path | None = None  # default: <venvs_dir>/<name>/bin/python
    env: Mapping[str, str] = field(default_factory=dict)
    version: str | None = None  # default: read from stages/<name>/pyproject.toml

    @property
    def project_dir(self) -> Path:
        return STAGES_DIR / self.name

    def python_path(self, config: Config) -> Path:
        return self.python or config.paths.venvs_dir / self.name / "bin" / "python"

    def resolved_version(self) -> str | None:
        if self.version is not None:
            return self.version
        if self.function is not None:
            return __version__
        pyproject = self.project_dir / "pyproject.toml"
        if not pyproject.is_file():
            return None
        with pyproject.open("rb") as f:
            return tomllib.load(f).get("project", {}).get("version")


_SEP = "02_separate"

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
