from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from app.inference.memory import process_rss_mb, reset_vram_peak, vram_peak_mb

log_header = (
    "# vram_mb: CUDA 为这一步的峰值显存；MPS 没有峰值计数，记步骤结束时的占用量。"
    "VLM 记 Ollama 报告的显存。\n"
    "# rss_mb: 当前 Python 进程常驻内存，不含 Ollama 进程。\n"
)


@dataclass
class StepMark:
    name: str
    seconds: float = 0.0
    vram_mb: float | None = None
    rss_mb: float | None = None
    detail: str = ""
    vram_override: float | None = None


class SceneProfile:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.rows: list[StepMark] = []
        self.started = time.perf_counter()

    @contextmanager
    def step(self, name: str):
        mark = StepMark(name)
        reset_vram_peak()
        started = time.perf_counter()
        try:
            yield mark
        finally:
            mark.seconds = round(time.perf_counter() - started, 2)
            mark.vram_mb = mark.vram_override if mark.vram_override is not None else vram_peak_mb()
            mark.rss_mb = process_rss_mb()
            self.rows.append(mark)

    def render(self) -> str:
        lines = [
            log_header.rstrip(),
            f"filename={self.filename}",
            f"{'step':<16}{'seconds':>10}{'vram_mb':>12}{'rss_mb':>12}  detail",
        ]
        for row in self.rows:
            lines.append(
                f"{row.name:<16}{row.seconds:>10.2f}{_num(row.vram_mb):>12}{_num(row.rss_mb):>12}  {row.detail}"
            )
        total = round(time.perf_counter() - self.started, 2)
        lines.append(f"{'total':<16}{total:>10.2f}")
        return "\n".join(lines) + "\n"

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path


def _num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"
