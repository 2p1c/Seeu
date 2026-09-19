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
log = logging.getLogger("roommind.yolo")


def _check_numpy() -> None:
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


def _device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


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
    def __init__(self, weights: str | Path | None = None) -> None:
        _check_numpy()
        import torch
        from ultralytics import YOLO

        weights_path = _ensure_weights(weights_path_for(weights))
        self.device = _device()
        log.info("loading YOLO weights=%s device=%s", weights_path, self.device)
        self.model = YOLO(str(weights_path))
        log.info("YOLO ready device=%s cuda=%s", self.device, torch.cuda.is_available())

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
        result = results[0]
        objects: list[DetectedObject] = []
        if result.boxes is not None:
            names = result.names
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().tolist()
                cls_id = int(box.cls[0].item())
                objects.append(
                    DetectedObject.model_validate(
                        {
                            "class": names[cls_id],
                            "bbox": [round(float(v), 1) for v in xyxy],
                        }
                    )
                )

        annotated = result.plot()
        detection = DetectionResult(
            timestamp=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            objects=objects,
            annotated_image=bgr_to_jpeg_b64(annotated) if include_image else None,
        )
        log.info(
            "detected %d objects: %s",
            len(objects),
            ", ".join(item.class_name for item in objects) or "(none)",
        )
        return detection, annotated
