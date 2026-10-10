"""report.md: a human summary of a job (see "Diretório do job" in docs/ARCHITECTURE.md)."""

from datetime import datetime
from pathlib import Path

from stagekit import atomic_write_text

from dublador.manifest import Manifest

REPORT_NAME = "report.md"


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    parse = lambda t: datetime.fromisoformat(t.replace("Z", "+00:00"))  # noqa: E731
    return (parse(end) - parse(start)).total_seconds()


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def render(manifest: Manifest) -> str:
    stages = manifest.data["stages"]
    total = sum(
        s for e in stages.values() if (s := _seconds(e.get("started_at"), e.get("finished_at")))
    )
    lines = [
        f"# dublador job `{manifest.job_id}`",
        "",
        f"- Video: `{manifest.video_path}`",
        f"- Output: `{manifest.output_path}`",
        f"- Time in stages (last run of each): {_duration(total)}",
        "",
        "| Stage | Status | Time | Warnings | Error |",
        "|---|---|---|---|---|",
    ]
    for name, entry in stages.items():
        seconds = _seconds(entry.get("started_at"), entry.get("finished_at"))
        error = (entry.get("error") or "").replace("|", "\\|")
        lines.append(
            f"| {name} | {entry['status']} | {_duration(seconds)} "
            f"| {len(entry.get('warnings', []))} | {error} |"
        )
    warned = [(name, e["warnings"]) for name, e in stages.items() if e.get("warnings")]
    if warned:
        lines += ["", "## Warnings", ""]
        for name, warnings in warned:
            lines += [f"### {name}", "", *(f"- {w}" for w in warnings), ""]
    return "\n".join(lines).rstrip() + "\n"


def write_report(manifest: Manifest, job_dir: Path) -> Path:
    path = job_dir / REPORT_NAME
    atomic_write_text(path, render(manifest))
    return path
