"""Progress events, one JSON object per line on stdout (read by the orchestrator)."""

import json
import sys
from typing import Any, TextIO


class Progress:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    def emit(self, event: str, **fields: Any) -> None:
        stream = self._stream or sys.stdout
        stream.write(json.dumps({"event": event, **fields}, ensure_ascii=False) + "\n")
        stream.flush()

    def progress(self, done: int, total: int) -> None:
        self.emit("progress", done=done, total=total)

    def log(self, message: str) -> None:
        self.emit("log", message=message)

    def warning(self, message: str) -> None:
        self.emit("warning", message=message)
