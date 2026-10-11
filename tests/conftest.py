import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from dublador import ffmpeg, stages
from dublador.stages import StageSpec

FAKE_STAGES = Path(__file__).parent / "fake_stages"

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg not installed",
)


@pytest.fixture(scope="session")
def video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3 s synthetic video with one AAC audio track."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("media") / "Sample Talk.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def fake_spec(
    name: str, module: str, inputs: tuple[str, ...], outputs: tuple[str, ...]
) -> StageSpec:
    return StageSpec(
        name,
        inputs=inputs,
        outputs=outputs,
        module=module,
        python=Path(sys.executable),
        env={"PYTHONPATH": str(FAKE_STAGES)},
        version="test",
    )


def _pipeline(fail: str | None = None) -> tuple[StageSpec, ...]:
    extract, mux = stages.PIPELINE[0], stages.PIPELINE[-1]
    sep = "02_separate"
    separate = fake_spec(
        "separate",
        "fake_separate",
        (ffmpeg.AUDIO_WAV,),
        (f"{sep}/vocals.wav", f"{sep}/accompaniment.wav"),
    )
    mix = fake_spec("mix", "fake_mix", (f"{sep}/accompaniment.wav",), (ffmpeg.DUB_WAV,))
    specs = (extract, separate, mix, mux)
    if fail:
        specs = tuple(
            StageSpec(**{**vars(s), "env": {**s.env, "FAKE_FAIL": fail}}) if s.module else s
            for s in specs
        )
    return specs


@pytest.fixture
def fake_pipeline() -> Callable[..., tuple[StageSpec, ...]]:
    """extract → fake separate → fake mix → mux; `fail=<stage>` forces a failure."""
    return _pipeline


@pytest.fixture
def make_fake_spec() -> Callable[..., StageSpec]:
    """fake_spec(name, module, inputs, outputs) for stages in tests/fake_stages."""
    return fake_spec
