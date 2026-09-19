from __future__ import annotations

from typing import Any

from app.yolo.detector import YOLODetector, bytes_to_bgr
from app.yolo.schema import DetectionResult


class YOLOService:
    def __init__(self) -> None:
        self._detector: YOLODetector | None = None

    def detector(self) -> YOLODetector:
        if self._detector is None:
            self._detector = YOLODetector()
        return self._detector

    def detect_image(self, image: bytes) -> DetectionResult:
        if not image:
            raise ValueError("empty image")
        try:
            frame = bytes_to_bgr(image)
        except Exception as exc:
            raise ValueError("cannot decode image") from exc
        detection, _ = self.detector().detect(frame, include_image=True)
        return detection

    def detect_frame(
        self,
        frame: Any,
        *,
        include_image: bool = False,
    ) -> tuple[DetectionResult, Any]:
        return self.detector().detect(frame, include_image=include_image)
