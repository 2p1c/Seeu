from __future__ import annotations

import base64
import logging
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from app.yolo.schema import DetectedObject, DetectionResult

WEIGHTS = "yolo26s.pt"
IMGSZ = 640
JPEG_QUALITY = 85
log = logging.getLogger("roommind.yolo")


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


class YOLODetector:
    def __init__(self, weights: str = WEIGHTS) -> None:
        import torch
        from ultralytics import YOLO

        self.device = _device()
        log.info("loading YOLO weights=%s device=%s", weights, self.device)
        self.model = YOLO(weights)
        log.info("YOLO ready device=%s cuda=%s", self.device, torch.cuda.is_available())

    def detect(
        self,
        image_bgr: Any,
        *,
        include_image: bool = True,
    ) -> tuple[DetectionResult, Any]:
        results = self.model.predict(
            source=image_bgr,
            device=self.device,
            imgsz=IMGSZ,
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
