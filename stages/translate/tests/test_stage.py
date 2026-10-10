"""The stage against a fake Ollama HTTP server."""

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from stagekit import main, read_json, write_json
from translate.__main__ import run

REQUESTS: list[tuple[str, dict]] = []


class FakeOllama(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        REQUESTS.append((self.path, {}))
        data = json.dumps({"models": []}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append((self.path, body))
        if self.path == "/api/chat":
            ids = [line["id"] for line in json.loads(body["messages"][1]["content"])["translate"]]
            content = json.dumps({"translations": [{"id": i, "text": f"olá {i}"} for i in ids]})
            reply = {"message": {"role": "assistant", "content": content}}
        else:
            reply = {"done": True}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def ollama_url() -> Iterator[str]:
    REQUESTS.clear()
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def make_job(tmp_path: Path, url: str) -> tuple[Path, Path]:
    job = tmp_path / "job"
    segments = [
        {
            "id": f"seg_{i:04d}",
            "start": i * 2.0,
            "end": i * 2.0 + 1.5,
            "text": f"line {i}",
            "speaker": None,
            "words": [],
        }
        for i in range(1, 4)
    ]
    write_json(
        job / "03_transcribe" / "transcript.json",
        {"schema": "transcript/v1", "language": "en", "segments": segments},
    )
    speakers = {"schema": "speakers/v1", "speakers": [{
        "id": "SPK_00", "speech_seconds": 4.5, "gender": "unknown", "gender_confidence": None,
        "strategy": "clone", "voice": None, "reference": None,
    }]}  # fmt: skip
    write_json(job / "05_speakers" / "speakers.json", speakers)
    config = tmp_path / "config.toml"
    config.write_text(
        f'[paths]\nmodels_dir = "/m"\n[translate]\nmodel = "fake"\nollama_url = "{url}"\n'
    )
    return job, config


def test_stage_writes_translation_and_unloads_model(tmp_path: Path, ollama_url: str) -> None:
    job, config = make_job(tmp_path, ollama_url)
    main("translate", run=run, argv=["--job-dir", str(job), "--config", str(config)])
    doc = read_json(job / "06_translate" / "translation.json", "translation/v1")
    assert [s["text"] for s in doc["segments"]] == ["Olá seg_0001", "Olá seg_0002", "Olá seg_0003"]
    assert doc["segments"][0]["speaker"] == "SPK_00"
    chat = [b for p, b in REQUESTS if p == "/api/chat"]
    assert chat[0]["think"] is False and chat[0]["format"]["required"] == ["translations"]
    assert REQUESTS[-2:] == [("/api/generate", {"model": "fake", "keep_alive": 0}), ("/api/ps", {})]

    # Second run: everything cached, no chat request, model still unloaded.
    REQUESTS.clear()
    main("translate", run=run, argv=["--job-dir", str(job), "--config", str(config)])
    assert [p for p, _ in REQUESTS] == ["/api/generate", "/api/ps"]

    # --only forces those ids again.
    REQUESTS.clear()
    main(
        "translate",
        run=run,
        argv=["--job-dir", str(job), "--config", str(config), "--only", "seg_0002"],
    )
    chat = [b for p, b in REQUESTS if p == "/api/chat"]
    assert len(chat) == 1


def test_unreachable_ollama_fails_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    job, config = make_job(tmp_path, "http://127.0.0.1:9")
    with pytest.raises(SystemExit) as exc:
        main("translate", run=run, argv=["--job-dir", str(job), "--config", str(config)])
    assert exc.value.code == 1
    assert "not reachable" in capsys.readouterr().err
