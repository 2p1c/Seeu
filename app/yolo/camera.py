from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2

from app.yolo.service import YOLOService

log = logging.getLogger("roommind.yolo")


def list_video_devices() -> list[tuple[str, str]]:
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return []
    devices: list[tuple[str, str]] = []
    for node in sorted(root.glob("video*")):
        name_file = node / "name"
        name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else "?"
        devices.append((f"/dev/{node.name}", name))
    return devices


def _format_devices() -> str:
    devices = list_video_devices()
    if not devices:
        return "(没有找到 /dev/video*)"
    return "\n".join(f"  {path}\t{name}" for path, name in devices)


def _parse_source(raw: str) -> int | str:
    return int(raw) if raw.isdigit() else raw


def _open_capture(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if cap.isOpened():
        return cap
    cap.release()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 source={source}。当前设备:\n{_format_devices()}"
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def run_camera(
    source: int | str,
    *,
    show: bool = False,
    save_dir: Path | None = None,
    max_fps: float = 5.0,
) -> None:
    if show and not _has_display():
        log.warning(
            "当前没有 DISPLAY，忽略 --show，避免 Qt 卡住。"
            "要在开发板屏幕上显示请用: DISPLAY=:0 python3 -m app.yolo --source %s --show"
            "；SSH 联调用: --save-dir /tmp/yolo-frames",
            source,
        )
        show = False

    service = YOLOService()
    cap = _open_capture(source)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    log.info("camera opened source=%s", source)
    min_interval = 0.0 if max_fps <= 0 else 1.0 / max_fps
    last = 0.0
    index = 0
    misses = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                misses += 1
                if misses >= 30:
                    log.warning("failed to read frame")
                    break
                time.sleep(0.05)
                continue
            misses = 0
            if index == 0:
                log.info("frame shape=%s dtype=%s", getattr(frame, "shape", None), frame.dtype)
            now = time.monotonic()
            if min_interval and now - last < min_interval:
                continue
            last = now
            result, annotated = service.detect_frame(frame, include_image=False)
            print(result.model_dump_json(by_alias=True, exclude={"annotated_image"}, indent=2), flush=True)
            if show:
                cv2.imshow("RoomMind YOLO", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if save_dir is not None:
                path = save_dir / f"frame_{index:06d}.jpg"
                cv2.imwrite(str(path), annotated)
            index += 1
    except KeyboardInterrupt:
        log.info("camera stopped")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用 USB 摄像头做 YOLO26s 检测，结果打印到终端")
    parser.add_argument("--source", default="0", help="摄像头编号或路径，例如 0 或 /dev/video0")
    parser.add_argument("--list", action="store_true", help="列出本机 /dev/video* 及名称后退出")
    parser.add_argument("--show", action="store_true", help="弹出标注画面窗口，按 q 退出")
    parser.add_argument("--save-dir", type=Path, default=None, help="把标注图存到该目录")
    parser.add_argument("--max-fps", type=float, default=5.0, help="终端打印上限，0 表示不限制")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.list:
        print(_format_devices())
        return 0
    run_camera(
        _parse_source(args.source),
        show=args.show,
        save_dir=args.save_dir,
        max_fps=args.max_fps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
