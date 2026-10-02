"""感知流水线的当前阶段。页面在 POST /api/scene 返回前轮询它。"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field


@dataclass
class _State:
    running: bool = False
    stage: str = "idle"
    detail: str = ""
    index: int | None = None
    total: int | None = None
    error: str = ""
    details: dict[str, str] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    open_stage: str | None = None
    open_started: float | None = None
    partial: dict = field(default_factory=dict)


_lock = threading.Lock()
_state = _State()


def begin() -> None:
    global _state
    with _lock:
        _state = _State(running=True, stage="sam")


def _close(now: float) -> None:
    if _state.open_stage and _state.open_started is not None:
        elapsed = now - _state.open_started
        _state.seconds[_state.open_stage] = _state.seconds.get(_state.open_stage, 0.0) + elapsed
        _state.open_started = None


def _seconds() -> dict[str, float]:
    seconds = dict(_state.seconds)
    if _state.open_stage and _state.open_started is not None:
        live = time.perf_counter() - _state.open_started
        seconds[_state.open_stage] = seconds.get(_state.open_stage, 0.0) + live
    return {name: round(value, 2) for name, value in seconds.items()}


def enter(stage: str) -> None:
    """开始给这个服务计时。同一个服务再次进入时接着累加，比如逐个物体的描述。"""
    now = time.perf_counter()
    with _lock:
        if _state.open_stage == stage and _state.open_started is not None:
            return
        if _state.open_stage == stage:
            _state.open_started = now
            _state.stage = stage
            _state.running = True
            return
        _close(now)
        _state.open_stage = stage
        _state.open_started = now
        _state.stage = stage
        _state.running = True
        _state.error = ""


def leave(stage: str) -> None:
    """停掉这个服务的计时。停表后的秒数不再增加。"""
    now = time.perf_counter()
    with _lock:
        if _state.open_stage != stage:
            return
        _close(now)


def publish(key: str, value) -> None:
    """记下这一阶段已经能给页面看的结果。后面的阶段失败时，这些结果还在。"""
    with _lock:
        _state.partial[key] = value


def mark(
    stage: str,
    detail: str = "",
    index: int | None = None,
    total: int | None = None,
) -> None:
    with _lock:
        _state.running = True
        _state.stage = stage
        _state.detail = detail
        _state.index = index
        _state.total = total
        _state.error = ""
        if detail:
            _state.details[stage] = detail


def finish(detail: str = "") -> None:
    with _lock:
        _close(time.perf_counter())
        _state.running = False
        _state.stage = "done"
        _state.detail = detail
        _state.index = None
        _state.total = None
        _state.error = ""
        if detail:
            _state.details["done"] = detail


def fail(message: str) -> None:
    with _lock:
        _close(time.perf_counter())
        _state.running = False
        _state.error = message


def snapshot() -> dict:
    with _lock:
        return {
            "running": _state.running,
            "stage": _state.stage,
            "detail": _state.detail,
            "index": _state.index,
            "total": _state.total,
            "error": _state.error,
            "details": dict(_state.details),
            "seconds": _seconds(),
            "partial": copy.deepcopy(_state.partial),
        }
