"""Stage 06 — translate (v0): EN to pt-BR with a local LLM through Ollama.

Windows of segments with surrounding context, JSON Schema output validated
and re-asked on errors. The model is always unloaded at the end so the TTS
stage gets the whole GPU. Syllable budgets, N candidates and the `shorten`
mode arrive in M3.
"""

import functools
from importlib import metadata
from pathlib import Path
from typing import Any

from stagekit import StageContext, StageError, main, read_json, write_json
from stagekit.probe import report

from translate.core import (
    RESPONSE_SCHEMA,
    Segment,
    TranslationError,
    input_hash,
    translate_all,
)
from translate.ollama import Ollama, OllamaError

VERSION = metadata.version("dublador-stage-translate")
DEFAULTS: dict[str, Any] = {
    "model": "gemma4:e4b-it-qat",
    "ollama_url": "http://localhost:11434",
    "batch_size": 12,
    "context_segments": 3,
    "retries": 3,
    "temperature": 0.2,
    "seed": 42,
    "num_ctx": 8192,
}


def settings(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("translate", {})
    unknown = sorted(set(section) - set(DEFAULTS))
    if unknown:
        raise StageError(f"unknown [translate] key(s): {', '.join(unknown)}")
    return {**DEFAULTS, **section}


def _previous(path: Path) -> dict[str, dict[str, Any]]:
    """Segments of an earlier translation.json, for the per-segment cache."""
    if not path.is_file():
        return {}
    try:
        return {s["id"]: s for s in read_json(path, "translation/v1")["segments"]}
    except Exception:  # an unreadable cache is just a cache miss
        return {}


def run(ctx: StageContext) -> None:
    opts = settings(ctx.config)
    languages = ctx.config.get("languages", {})
    source, target = languages.get("source", "en"), languages.get("target", "pt-BR")
    transcript = read_json(ctx.stage_dir("transcribe") / "transcript.json", "transcript/v1")
    speakers = read_json(ctx.stage_dir("speakers") / "speakers.json", "speakers/v1")
    default_speaker = speakers["speakers"][0]["id"] if speakers["speakers"] else "SPK_00"

    segments = [
        Segment(s["id"], s["start"], s["end"], s.get("speaker") or default_speaker, s["text"])
        for s in transcript["segments"]
    ]
    hashes = {
        s.id: input_hash(s, model=opts["model"], source=source, target=target) for s in segments
    }
    out_path = ctx.out_dir / "translation.json"
    previous = _previous(out_path)
    known = {
        i: p["text"] for i, p in previous.items() if i in hashes and p["input_hash"] == hashes[i]
    }
    todo = {s.id for s in segments if ctx.selected(s.id) and s.id not in known}
    if ctx.only is not None:  # forced re-translation of the listed ids
        todo = {s.id for s in segments if ctx.selected(s.id)}
    ctx.progress.log(f"{len(todo)} to translate, {len(segments) - len(todo)} cached")

    client = Ollama(opts["ollama_url"], opts["model"])
    chat = functools.partial(
        client.chat,
        schema=RESPONSE_SCHEMA,
        options={
            "temperature": opts["temperature"],
            "seed": opts["seed"],
            "num_ctx": opts["num_ctx"],
        },
    )
    try:
        texts = translate_all(
            chat,
            segments,
            todo=todo,
            known=known,
            source=source,
            target=target,
            batch_size=opts["batch_size"],
            context=opts["context_segments"],
            retries=opts["retries"],
            on_progress=ctx.progress.progress,
        )
    except (TranslationError, OllamaError) as e:
        raise StageError(str(e)) from e
    finally:
        try:
            client.unload()
        except OllamaError as e:
            ctx.progress.warning(f"could not unload {opts['model']}: {e}")

    doc = {
        "schema": "translation/v1",
        "model": opts["model"],
        "source_language": source,
        "target_language": target,
        "segments": [
            {
                "id": s.id,
                "start": s.start,
                "end": s.end,
                "speaker": s.speaker,
                "source_text": s.text,
                "text": texts[s.id],
                "input_hash": hashes[s.id],
            }
            for s in segments
        ],
    }
    write_json(out_path, doc)


def probe() -> dict[str, Any]:
    return report(VERSION, gpu=False)


if __name__ == "__main__":
    main("translate", run=run, probe=probe)
