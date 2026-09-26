from __future__ import annotations

import fcntl
import logging
import os
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("roommind.trt")


class GpuBusy(Exception):
    pass


def lock_path() -> Path:
    return Path(os.environ.get("ROOMIND_GPU_LOCK", "/tmp/roomind-gpu.lock"))


class GpuLease:
    """跨进程互斥。锁在进程退出时由系统释放，感知服务每次请求拿一次。"""

    def __init__(self, holder: str, timeout_s: float = 5) -> None:
        self.holder = holder
        self.timeout_s = timeout_s
        self._fd = None

    def __enter__(self) -> "GpuLease":
        path = lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = path.open("a+")
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                owner = _read_holder(self._fd)
                if time.monotonic() >= deadline:
                    self._fd.close()
                    self._fd = None
                    raise GpuBusy(
                        f"GPU 正被 {owner} 占用。8GB 统一内存上摄像头跟踪和感知推理要轮流进行，"
                        "停掉 roomind@yolo 或等当前请求结束再试。"
                    ) from None
                time.sleep(0.05)
        self._fd.seek(0)
        self._fd.truncate()
        self._fd.write(self.holder)
        self._fd.flush()
        log.info("gpu lease acquired by %s", self.holder)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            self._fd.close()
            self._fd = None
        log.info("gpu lease released by %s", self.holder)


def _read_holder(fd) -> str:
    fd.seek(0)
    text = fd.read().strip()
    fd.seek(0)
    return text or "另一个进程"


def release_remote() -> None:
    """让感知进程卸掉已经加载的模型。调用方需要先拿到 GpuLease，避免和请求交错。"""
    url = os.environ.get("ROOMIND_API", "http://127.0.0.1:8000") + "/api/deploy/release"
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    except Exception as exc:
        log.info("perception release skipped: %s", exc)
        return
    log.info("perception released loaded models")
