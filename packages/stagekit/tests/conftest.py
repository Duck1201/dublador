"""Minimal valid documents for every schema."""

import copy
from typing import Any

import pytest

H = "sha256:" + "0" * 64
T = "2026-10-10T12:00:00Z"

EXAMPLES: dict[str, dict[str, Any]] = {
    "transcript/v1": {
        "schema": "transcript/v1",
        "language": "en",
        "segments": [
            {
                "id": "seg_0001",
                "start": 0.5,
                "end": 1.8,
                "text": "Hello there.",
                "speaker": None,
                "words": [
                    {"word": "Hello", "start": 0.5, "end": 0.9, "score": 0.98},
                    {"word": "there."},
                ],
            }
        ],
    },
    "speakers/v1": {
        "schema": "speakers/v1",
        "speakers": [
            {
                "id": "SPK_00",
                "speech_seconds": 42.0,
                "gender": "unknown",
                "gender_confidence": None,
                "strategy": "clone",
                "voice": None,
                "reference": {
                    "path": "05_speakers/refs/SPK_00.wav",
                    "start": 3.0,
                    "end": 12.5,
                    "text": "Reference text.",
                },
            }
        ],
    },
    "translation/v1": {
        "schema": "translation/v1",
        "model": "gemma4:e4b-it-qat",
        "source_language": "en",
        "target_language": "pt-BR",
        "segments": [
            {
                "id": "seg_0001",
                "start": 0.5,
                "end": 1.8,
                "speaker": "SPK_00",
                "source_text": "Hello there.",
                "text": "Olá.",
                "input_hash": H,
            }
        ],
    },
    "synthesis/v1": {
        "schema": "synthesis/v1",
        "backend": "qwen3_tts",
        "sample_rate": 24000,
        "segments": [
            {
                "id": "seg_0001",
                "path": "07_synthesize/segments/seg_0001.wav",
                "duration": 1.2,
                "input_hash": H,
            }
        ],
    },
    "manifest/v1": {
        "schema": "manifest/v1",
        "job_id": "talk-1a2b3c4d",
        "video": {"path": "/videos/talk.mp4", "hash": H, "size": 1024},
        "created_at": T,
        "stages": {
            "extract": {
                "status": "done",
                "started_at": T,
                "finished_at": T,
                "version": "0.1.0",
                "model": None,
                "inputs_hash": H,
                "error": None,
            },
            "separate": {"status": "pending"},
        },
    },
    "stage-probe/v1": {
        "schema": "stage-probe/v1",
        "stage": "separate",
        "version": "0.1.0",
        "python": "3.12.11",
        "packages": {"torch": "2.11.0+rocm7.2"},
        "gpu": {"available": True, "name": "AMD Radeon RX 7600", "arch_list": ["gfx1102"]},
    },
}


@pytest.fixture
def examples() -> dict[str, dict[str, Any]]:
    """Fresh copies of the example documents, safe to mutate."""
    return copy.deepcopy(EXAMPLES)
