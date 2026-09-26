from __future__ import annotations

from pathlib import Path
from typing import Any

from app.inference.trt.budget import claim_gpu
from app.yolo.detector import DEFAULT_IMGSZ, YOLODetector, bytes_to_bgr, save_annotated_jpeg
from app.yolo.schema import DetectionResult


class YOLOService:
    def __init__(self, model: str | None = None, tracker: str | None = None) -> None:
        self._model = model
        self._tracker = tracker
        self._detector: YOLODetector | None = None

    def detector(self) -> YOLODetector:
        if self._detector is None:
            self._detector = YOLODetector(self._model, tracker=self._tracker)
        return self._detector

    def detect_image(
        self,
        image: bytes,
        filename: str | None = None,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> DetectionResult:
        if not image:
            raise ValueError("empty image")
        try:
            frame = bytes_to_bgr(image)
        except Exception as exc:
            raise ValueError("cannot decode image") from exc
        claim_gpu("yolo", self.release)
        detection, annotated = self.detector().detect(
            frame,
            include_image=True,
            imgsz=imgsz,
        )
        stem = Path(filename or "yolo-annotated").stem or "yolo-annotated"
        detection.saved_path = str(save_annotated_jpeg(annotated, stem))
        return detection

    def detect_frame(
        self,
        frame: Any,
        *,
        include_image: bool = False,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> tuple[DetectionResult, Any]:
        claim_gpu("yolo", self.release)
        return self.detector().detect(frame, include_image=include_image, imgsz=imgsz)

    def track_frame(
        self,
        frame: Any,
        *,
        include_image: bool = False,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> tuple[DetectionResult, Any]:
        claim_gpu("yolo", self.release)
        return self.detector().track(frame, include_image=include_image, imgsz=imgsz)

    def release(self) -> None:
        detector = self._detector
        self._detector = None
        if detector is not None:
            detector.release()
