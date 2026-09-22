from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2

log = logging.getLogger("roommind.camera")


def _is_linux() -> bool:
    return sys.platform.startswith("linux")


def list_video_devices() -> list[tuple[str, str]]:
    root = Path("/sys/class/video4linux")
    if _is_linux() and root.is_dir():
        devices: list[tuple[str, str]] = []
        for node in sorted(root.glob("video*")):
            name_file = node / "name"
            name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else "?"
            devices.append((f"/dev/{node.name}", name))
        return devices
    found: list[tuple[str, str]] = []
    for index in range(5):
        cap = cv2.VideoCapture(index)
        opened = cap.isOpened()
        cap.release()
        if opened:
            found.append((str(index), "camera"))
    return found


def format_devices() -> str:
    devices = list_video_devices()
    if not devices:
        return "(没有找到摄像头)"
    return "\n".join(f"  {path}\t{name}" for path, name in devices)


def parse_source(raw: str) -> int | str:
    return int(raw) if raw.isdigit() else raw


def _set_capture_size(cap: cv2.VideoCapture, width: int, height: int) -> None:
    if _is_linux():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("camera resolution requested=%dx%d actual=%dx%d", width, height, actual_w, actual_h)
    if (actual_w, actual_h) != (width, height):
        hint = "可执行: v4l2-ctl --list-formats-ext -d /dev/video0" if _is_linux() else "可把 --width/--height 改成摄像头实际分辨率"
        log.warning(
            "摄像头未接受 %dx%d，当前是 %dx%d。%s",
            width,
            height,
            actual_w,
            actual_h,
            hint,
        )


def open_capture(
    source: int | str,
    width: int = 1920,
    height: int = 1080,
) -> cv2.VideoCapture:
    if isinstance(source, int) and _is_linux():
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 source={source}。当前设备:\n{format_devices()}"
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _set_capture_size(cap, width, height)
    log.info("camera opened source=%s", source)
    return cap


def has_display() -> bool:
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def camera_props(cap: cv2.VideoCapture) -> dict[str, int | float | str]:
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    fourcc = "".join(ch for ch in fourcc if ch.isprintable()).strip()
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(float(cap.get(cv2.CAP_PROP_FPS) or 0), 2),
        "fourcc": fourcc or "unknown",
    }


def iter_frames(cap: cv2.VideoCapture, *, max_fps: float = 5.0) -> Iterator:
    min_interval = 0.0 if max_fps <= 0 else 1.0 / max_fps
    last = 0.0
    misses = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            misses += 1
            if misses >= 30:
                log.warning("failed to read frame")
                return
            time.sleep(0.05)
            continue
        misses = 0
        now = time.monotonic()
        if min_interval and now - last < min_interval:
            continue
        last = now
        yield frame
