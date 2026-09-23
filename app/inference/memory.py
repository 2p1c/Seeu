from __future__ import annotations

import gc
from pathlib import Path


def accelerator() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def inference_dtype(device: str):
    import torch

    if device == "cuda":
        return torch.float16
    return torch.float32


def is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def reset_vram_peak() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def vram_peak_mb() -> float | None:
    """CUDA 为本次统计区间的峰值。MPS 没有峰值计数，返回当前占用量。"""
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    mps = getattr(torch, "mps", None)
    if mps is not None and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return round(mps.current_allocated_memory() / (1024**2), 1)
    return None


def process_rss_mb() -> float | None:
    import os
    import sys

    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
        except (OSError, ValueError):
            return None
        return None
    try:
        import subprocess

        raw = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
        ).strip()
        return round(int(raw) / 1024, 1)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def release_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    empty = getattr(mps, "empty_cache", None) if mps is not None else None
    if empty is not None:
        try:
            empty()
        except Exception:
            pass
    gc.collect()
