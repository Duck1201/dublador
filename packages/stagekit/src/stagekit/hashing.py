"""Stable hashes of stage inputs, used for per-stage and per-segment caching."""

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def hash_value(value: Any) -> str:
    """SHA-256 of a JSON-serializable value, independent of dict key order."""
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
