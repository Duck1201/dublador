"""In-process ffmpeg stages: 01 extract and 10 mux (docs/ARCHITECTURE.md)."""

import json
import subprocess
from pathlib import Path
from typing import Any

from stagekit import atomic_path, atomic_write_json

from dublador.config import Config
from dublador.manifest import Manifest

AUDIO_WAV = "01_extract/audio.wav"
PROBE_JSON = "01_extract/probe.json"
DUB_WAV = "09_mix/dub.wav"

# ISO 639-2 codes for track metadata; unknown languages get no tag.
_ISO_639_2 = {"en": "eng", "pt": "por", "es": "spa", "fr": "fra", "de": "deu", "it": "ita"}
# Audio codecs MP4 can carry as-is; anything else is re-encoded to AAC.
_MP4_AUDIO = {"aac", "mp3", "ac3", "eac3", "alac"}
_FFMPEG = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]


class FfmpegError(Exception):
    pass


def _run(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise FfmpegError(f"{cmd[0]} not found; install ffmpeg") from e
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise FfmpegError(f"{cmd[0]} failed (exit {result.returncode}): {tail}")
    return result.stdout


def ffprobe(video: Path) -> dict[str, Any]:
    out = _run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)]
    )
    return json.loads(out)


def select_audio_track(probe: dict[str, Any], audio_track: int | None) -> int:
    """Index among the audio streams: the requested one, else the default, else the first."""
    streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise FfmpegError("the video has no audio track")
    if audio_track is not None:
        if audio_track >= len(streams):
            raise FfmpegError(f"--audio-track {audio_track} out of range (0-{len(streams) - 1})")
        return audio_track
    for i, stream in enumerate(streams):
        if stream.get("disposition", {}).get("default"):
            return i
    return 0


def _audio_codec(probe: dict[str, Any], index: int) -> str:
    streams = [s for s in probe["streams"] if s.get("codec_type") == "audio"]
    return streams[index].get("codec_name", "")


def _language(code: str) -> str | None:
    return _ISO_639_2.get(code.split("-")[0].lower())


def extract(job_dir: Path, manifest: Manifest, config: Config) -> None:
    """Probe the video and extract the chosen audio track as PCM WAV."""
    probe = ffprobe(manifest.video_path)
    track = select_audio_track(probe, manifest.audio_track)
    atomic_write_json(job_dir / PROBE_JSON, probe)
    with atomic_path(job_dir / AUDIO_WAV) as tmp:
        _run(
            [
                *_FFMPEG,
                "-i", str(manifest.video_path),
                "-map", f"0:a:{track}", "-vn", "-sn", "-dn",
                "-c:a", "pcm_s16le",
                str(tmp),
            ]
        )  # fmt: skip


def mux_command(
    video: Path,
    dub: Path,
    output: Path,
    *,
    probe: dict[str, Any],
    audio_track: int,
    config: Config,
) -> list[str]:
    """Video copied as-is; dubbed track first and default; original track second."""
    original_codec = _audio_codec(probe, audio_track)
    copy_original = config.output.format == "mkv" or original_codec in _MP4_AUDIO
    cmd = [
        *_FFMPEG,
        "-i", str(video),
        "-i", str(dub),
        "-map", "0:v?", "-map", "1:a:0", "-map", f"0:a:{audio_track}",
        "-c:v", "copy",
        "-c:a:0", "aac", "-b:a:0", "192k",
        "-c:a:1", "copy" if copy_original else "aac",
    ]  # fmt: skip
    if not copy_original:
        cmd += ["-b:a:1", "192k"]
    tracks = [
        (config.languages.target, f"Dubbed ({config.languages.target})", "default"),
        (config.languages.source, f"Original ({config.languages.source})", "0"),
    ]
    for i, (lang, title, disposition) in enumerate(tracks):
        if iso := _language(lang):
            cmd += [f"-metadata:s:a:{i}", f"language={iso}"]
        cmd += [f"-metadata:s:a:{i}", f"title={title}", f"-disposition:a:{i}", disposition]
    if config.output.format == "mp4":
        cmd += ["-movflags", "+faststart"]
    return [*cmd, str(output)]


def mux(job_dir: Path, manifest: Manifest, config: Config) -> None:
    """Write the final video next to the source (or to --output)."""
    probe = json.loads((job_dir / PROBE_JSON).read_text())
    track = select_audio_track(probe, manifest.audio_track)
    with atomic_path(manifest.output_path) as tmp:
        _run(
            mux_command(
                manifest.video_path,
                job_dir / DUB_WAV,
                tmp,
                probe=probe,
                audio_track=track,
                config=config,
            )
        )
