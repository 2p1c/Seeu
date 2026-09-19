from __future__ import annotations

import base64
import logging
import urllib.request
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from app.yolo.schema import DetectedObject, DetectionResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "yolo26s.pt"
WEIGHTS_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
SAVE_DIR = PROJECT_ROOT / "test" / "tmp"
MIN_WEIGHTS_BYTES = 5_000_000
DEFAULT_IMGSZ = 640
JPEG_QUALITY = 85
TRACK_LOST_SEC = 1.0
log = logging.getLogger("roommind.yolo")


def _is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").is_file()


def _check_numpy() -> None:
    if not _is_jetson():
        return
    import numpy as np

    major = int(np.__version__.split(".", 1)[0])
    if major >= 2:
        raise RuntimeError(
            "Jetson 上的 PyTorch/OpenCV 需要 numpy 1.x，当前是 "
            f"{np.__version__}。请执行: pip install 'numpy>=1.23.5,<2'"
        )


def weights_path_for(model: str | Path | None = None) -> Path:
    raw = Path(model) if model else Path(DEFAULT_MODEL)
    name = raw.name if raw.suffix == ".pt" else f"{raw.name}.pt"
    return PROJECT_ROOT / "models" / name


def _ensure_weights(path: Path) -> Path:
    if path.is_file() and path.stat().st_size >= MIN_WEIGHTS_BYTES:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{WEIGHTS_BASE_URL}/{path.name}"
    log.info("downloading %s -> %s", url, path)
    tmp = path.with_suffix(".pt.part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"无法下载 {path.name}。请在项目根目录手动执行:\n"
            f"  mkdir -p models && wget -O models/{path.name} {url}"
        ) from exc
    if not tmp.is_file() or tmp.stat().st_size < MIN_WEIGHTS_BYTES:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"下载的 {path.name} 不完整。请手动执行:\n"
            f"  mkdir -p models && wget -O models/{path.name} {url}"
        )
    tmp.replace(path)
    return path


def write_bytetrack_config(frame_rate: float) -> Path:
    """ByteTrack 的 max_time_lost = frame_rate / 30 * track_buffer。

    track_buffer=30 时，frame_rate 取实际处理 FPS，丢失保留约 1 秒。
    """
    fps = max(1, int(round(frame_rate))) if frame_rate and frame_rate > 0 else 30
    path = PROJECT_ROOT / "models" / "bytetrack.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "tracker_type: bytetrack\n"
        "track_high_thresh: 0.5\n"
        "track_low_thresh: 0.1\n"
        "new_track_thresh: 0.6\n"
        "track_buffer: 30\n"
        "match_thresh: 0.8\n"
        "fuse_score: True\n"
        f"frame_rate: {fps}\n",
        encoding="utf-8",
    )
    log.info("ByteTrack config %s frame_rate=%d lost≈%.1fs", path, fps, TRACK_LOST_SEC)
    return path


def _track_id(box: Any) -> int | None:
    tid = getattr(box, "id", None)
    if tid is None:
        return None
    value = tid.view(-1)[0].item() if hasattr(tid, "view") else tid
    return int(value)


def _objects_from_result(result: Any) -> list[DetectedObject]:
    objects: list[DetectedObject] = []
    if result.boxes is None:
        return objects
    names = result.names
    for box in result.boxes:
        xyxy = box.xyxy[0].cpu().tolist()
        cls_id = int(box.cls[0].item())
        payload: dict[str, Any] = {
            "class": names[cls_id],
            "bbox": [round(float(v), 1) for v in xyxy],
        }
        track_id = _track_id(box)
        if track_id is not None:
            payload["track_id"] = track_id
        objects.append(DetectedObject.model_validate(payload))
    return objects


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def bytes_to_bgr(data: bytes) -> Any:
    import cv2
    import numpy as np
    from PIL import Image

    image = Image.open(BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def bgr_to_jpeg_b64(image_bgr: Any) -> str:
    import cv2

    ok, buf = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not ok:
        raise RuntimeError("failed to encode annotated jpeg")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def save_annotated_jpeg(image_bgr: Any, stem: str = "yolo-annotated") -> Path:
    import cv2

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_") or "yolo-annotated"
    path = SAVE_DIR / f"{safe}.jpg"
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"failed to write {path}")
    log.info("saved annotated image %s", path)
    return path


class YOLODetector:
    def __init__(
        self,
        weights: str | Path | None = None,
        tracker: str | Path | None = None,
    ) -> None:
        _check_numpy()
        import torch
        from ultralytics import YOLO

        weights_path = _ensure_weights(weights_path_for(weights))
        self.device = _device()
        self.tracker = str(tracker) if tracker else None
        log.info("loading YOLO weights=%s device=%s tracker=%s", weights_path, self.device, self.tracker)
        self.model = YOLO(str(weights_path))
        log.info("YOLO ready device=%s cuda=%s", self.device, torch.cuda.is_available())

    def _pack(
        self,
        result: Any,
        *,
        include_image: bool,
    ) -> tuple[DetectionResult, Any]:
        objects = _objects_from_result(result)
        annotated = result.plot()
        detection = DetectionResult(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            objects=objects,
            annotated_image=bgr_to_jpeg_b64(annotated) if include_image else None,
        )
        log.info(
            "detected %d objects: %s",
            len(objects),
            ", ".join(
                f"{item.class_name}#{item.track_id}" if item.track_id is not None else item.class_name
                for item in objects
            )
            or "(none)",
        )
        return detection, annotated

    def detect(
        self,
        image_bgr: Any,
        *,
        include_image: bool = True,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> tuple[DetectionResult, Any]:
        results = self.model.predict(
            source=image_bgr,
            device=self.device,
            imgsz=imgsz,
            verbose=False,
        )
        return self._pack(results[0], include_image=include_image)

    def track(
        self,
        image_bgr: Any,
        *,
        include_image: bool = False,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> tuple[DetectionResult, Any]:
        kwargs: dict[str, Any] = {
            "source": image_bgr,
            "device": self.device,
            "imgsz": imgsz,
            "verbose": False,
            "persist": True,
        }
        if self.tracker:
            kwargs["tracker"] = self.tracker
        results = self.model.track(**kwargs)
        return self._pack(results[0], include_image=include_image)
