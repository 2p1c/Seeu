from __future__ import annotations

from pathlib import Path
from typing import Any

from app.yolo.detector import YOLODetector, bytes_to_bgr, save_annotated_jpeg
from app.yolo.schema import DetectionResult


class YOLOService:
    def __init__(self) -> None:
        self._detector: YOLODetector | None = None

    def detector(self) -> YOLODetector:
        if self._detector is None:
            self._detector = YOLODetector()
        return self._detector

    def detect_image(self, image: bytes, filename: str | None = None) -> DetectionResult:
        if not image:
            raise ValueError("empty image")
        try:
            frame = bytes_to_bgr(image)
        except Exception as exc:
            raise ValueError("cannot decode image") from exc
        detection, annotated = self.detector().detect(frame, include_image=True)
        stem = Path(filename or "yolo-annotated").stem or "yolo-annotated"
        detection.saved_path = str(save_annotated_jpeg(annotated, stem))
        return detection

    def detect_frame(
        self,
        frame: Any,
        *,
        include_image: bool = False,
    ) -> tuple[DetectionResult, Any]:
        return self.detector().detect(frame, include_image=include_image)
