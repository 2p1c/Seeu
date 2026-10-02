from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger("roommind.trt")

_holder: tuple[str, Callable[[], None]] | None = None


def claim_gpu(name: str, release: Callable[[], None], *, unload_ollama: bool = True) -> None:
    """同一进程里只留一个占 GPU 的模型。再次声明同一个名字时什么也不卸。"""
    global _holder
    if _holder and _holder[0] == name:
        _holder = (name, release)
        return
    if _holder:
        log.info("gpu slot %s -> %s", _holder[0], name)
        _holder[1]()
    if unload_ollama:
        from app.inference.vlm.adapter import unload_model

        unload_model()
    _holder = (name, release)


def drop_claims() -> None:
    global _holder
    _holder = None


def current_claim() -> str | None:
    return _holder[0] if _holder else None
