"""Stage 06 — translate (v0): EN to pt-BR with a local LLM through Ollama.

Windows of segments with surrounding context, JSON Schema output validated
and re-asked on errors. Each line gets a syllable budget from its timing window;
the model answers 3 candidates and the longest one within the budget wins. With
--only, ids that the fit stage marked with `shorten_to` are rewritten shorter
from their current translation instead of being translated again (the M3 fit
loop). The model is always unloaded at the end so the TTS stage gets the GPU.
"""

import dataclasses
import functools
from importlib import metadata
from pathlib import Path
from typing import Any

from stagekit import StageContext, StageError, main, read_json, write_json
from stagekit.probe import report

from translate import syllables
from translate.core import (
    RESPONSE_SCHEMA,
    Segment,
    TranslationError,
    budgets,
    input_hash,
    translate_all,
    with_genders,
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
    # pt-BR speech rate of the TTS, counted with translate.syllables; calibrated
    # with OmniVoice on the reference video (docs/MODELS.md).
    "syllables_per_second": 5.7,
    # Silence kept before the next line (same as [fit] gap_seconds).
    "gap_seconds": 0.15,
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


def _shorten_targets(ctx: StageContext) -> dict[str, int]:
    """Syllable targets the fit stage asked for, by segment id."""
    path = ctx.stage_dir("fit") / "fit.json"
    if not path.is_file():
        return {}
    doc = read_json(path, "fit/v1")
    return {s["id"]: s["shorten_to"] for s in doc["segments"] if s.get("shorten_to")}


def run(ctx: StageContext) -> None:
    opts = settings(ctx.config)
    languages = ctx.config.get("languages", {})
    source, target = languages.get("source", "en"), languages.get("target", "pt-BR")
    transcript = read_json(ctx.stage_dir("diarize") / "transcript.json", "transcript/v1")
    speakers = read_json(ctx.stage_dir("speakers") / "speakers.json", "speakers/v1")
    default_speaker = speakers["speakers"][0]["id"] if speakers["speakers"] else "SPK_00"

    genders = {spk["id"]: spk["gender"] for spk in speakers["speakers"]}
    segments = with_genders(
        [
            Segment(s["id"], s["start"], s["end"], s.get("speaker") or default_speaker, s["text"])
            for s in transcript["segments"]
        ],
        genders,
    )
    limits = budgets(
        segments, syllables_per_second=opts["syllables_per_second"], gap=opts["gap_seconds"]
    )
    segments = [dataclasses.replace(s, budget=limits[s.id]) for s in segments]
    hashes = {
        s.id: input_hash(s, model=opts["model"], source=source, target=target) for s in segments
    }
    out_path = ctx.out_dir / "translation.json"
    previous = _previous(out_path)
    known = {
        i: p["text"] for i, p in previous.items() if i in hashes and p["input_hash"] == hashes[i]
    }
    todo = {s.id for s in segments if ctx.selected(s.id) and s.id not in known}
    shorten: dict[str, int] = {}
    if ctx.only is not None:
        todo = {s.id for s in segments if ctx.selected(s.id)}
        shorten = {i: n for i, n in _shorten_targets(ctx).items() if i in todo and i in previous}
    if shorten:  # rewrite the current translation shorter instead of translating again
        segments = [
            dataclasses.replace(s, budget=shorten[s.id], shorten=previous[s.id]["text"])
            if s.id in shorten
            else s
            for s in segments
        ]
        # The rewrite stays valid for the same inputs: keep the cache key.
        hashes.update({i: previous[i]["input_hash"] for i in shorten})
    ctx.progress.log(
        f"{len(todo)} to translate ({len(shorten)} to shorten), {len(segments) - len(todo)} cached"
    )

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
                "syllables": syllables.count(texts[s.id]),
                "budget": s.budget,
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
