from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from app.camera import format_devices, has_display, iter_frames, open_capture, parse_source
from app.inference.trt.lease import GpuLease, release_remote
from app.yolo.detector import DEFAULT_IMGSZ, DEFAULT_MODEL, write_bytetrack_config
from app.yolo.service import YOLOService

log = logging.getLogger("roommind.yolo")


def run_camera(
    source: int | str,
    *,
    show: bool = False,
    save_dir: Path | None = None,
    max_fps: float = 5.0,
    imgsz: int = DEFAULT_IMGSZ,
    width: int = 1920,
    height: int = 1080,
    model: str = DEFAULT_MODEL,
) -> None:
    if show and not has_display():
        log.warning(
            "当前没有 DISPLAY，忽略 --show，避免 Qt 卡住。"
            "要在开发板屏幕上显示请用: DISPLAY=:0 python3 -m app.yolo --source %s --show"
            "；SSH 联调用: --save-dir /tmp/yolo-frames",
            source,
        )
        show = False

    service = YOLOService(model, tracker=str(write_bytetrack_config(max_fps)))
    cap = open_capture(source, width=width, height=height)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    try:
        for frame in iter_frames(cap, max_fps=max_fps):
            if index == 0:
                log.info("frame shape=%s dtype=%s", getattr(frame, "shape", None), frame.dtype)
            result, annotated = service.track_frame(frame, include_image=False, imgsz=imgsz)
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
    parser = argparse.ArgumentParser(description="用 USB 摄像头做 YOLO26 检测，结果打印到终端")
    parser.add_argument("--source", default="0", help="摄像头编号或路径，例如 0 或 /dev/video0")
    parser.add_argument("--list", action="store_true", help="列出本机摄像头后退出")
    parser.add_argument("--show", action="store_true", help="弹出标注画面窗口，按 q 退出")
    parser.add_argument("--save-dir", type=Path, default=None, help="把标注图存到该目录")
    parser.add_argument("--max-fps", type=float, default=5.0, help="终端打印上限，0 表示不限制")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="YOLO LetterBox 输入边长，默认 640")
    parser.add_argument("--width", type=int, default=1920, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=1080, help="摄像头采集高度")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="权重名，例如 yolo26s / yolo26x，默认 yolo26s.pt",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.list:
        print(format_devices())
        return 0
    with GpuLease("yolo-camera", timeout_s=60):
        release_remote()
        run_camera(
            parse_source(args.source),
            show=args.show,
            save_dir=args.save_dir,
            max_fps=args.max_fps,
            imgsz=args.imgsz,
            width=args.width,
            height=args.height,
            model=args.model,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
