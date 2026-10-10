from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dublador.config import load_config
from dublador.ffmpeg import FfmpegError, mux_command, select_audio_track


def probe(*codecs: str, default: int | None = None) -> dict[str, Any]:
    streams: list[dict[str, Any]] = [{"codec_type": "video", "codec_name": "h264"}]
    for i, codec in enumerate(codecs):
        streams.append(
            {
                "codec_type": "audio",
                "codec_name": codec,
                "disposition": {"default": int(i == default)},
            }
        )
    return {"streams": streams}


def test_select_audio_track() -> None:
    assert select_audio_track(probe("aac", "aac"), None) == 0
    assert select_audio_track(probe("aac", "aac", default=1), None) == 1
    assert select_audio_track(probe("aac", "aac", default=1), 0) == 0
    with pytest.raises(FfmpegError, match="out of range"):
        select_audio_track(probe("aac"), 1)
    with pytest.raises(FfmpegError, match="no audio"):
        select_audio_track({"streams": []}, None)


def config(fmt: str, home: Path):
    return load_config({"output.format": fmt}, env={"HOME": str(home)}, cwd=home)


def args(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == flag]


def test_mkv_copies_original_track_and_tags_languages(tmp_path: Path) -> None:
    cmd = mux_command(
        Path("v.mkv"), Path("dub.wav"), Path("o.mkv"),
        probe=probe("opus"), audio_track=0, config=config("mkv", tmp_path),
    )  # fmt: skip
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert cmd[cmd.index("-c:a:1") + 1] == "copy"
    assert args(cmd, "-metadata:s:a:0") == ["language=por", "title=Dubbed (pt-BR)"]
    assert args(cmd, "-metadata:s:a:1") == ["language=eng", "title=Original (en)"]
    assert args(cmd, "-disposition:a:0") == ["default"]
    assert cmd[-1] == "o.mkv"


def test_mp4_reencodes_unsupported_original_codec(tmp_path: Path) -> None:
    cfg = config("mp4", tmp_path)
    opus = mux_command(
        Path("v"), Path("d"), Path("o.mp4"), probe=probe("opus"), audio_track=0, config=cfg
    )
    assert opus[opus.index("-c:a:1") + 1] == "aac"
    aac = mux_command(
        Path("v"), Path("d"), Path("o.mp4"), probe=probe("aac"), audio_track=0, config=cfg
    )
    assert aac[aac.index("-c:a:1") + 1] == "copy"
    assert "+faststart" in aac


def test_unknown_language_gets_no_tag(tmp_path: Path) -> None:
    cfg = config("mkv", tmp_path)
    cfg = replace(cfg, languages=replace(cfg.languages, target="xx"))
    cmd = mux_command(
        Path("v"), Path("d"), Path("o"), probe=probe("aac"), audio_track=0, config=cfg
    )
    assert args(cmd, "-metadata:s:a:0") == ["title=Dubbed (xx)"]
