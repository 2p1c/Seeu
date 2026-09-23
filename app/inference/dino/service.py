from __future__ import annotations

from app.inference.dino.detector import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_MAX_SIZE,
    DEFAULT_TEXT_THRESHOLD,
    DinoDetector,
    bytes_to_rgb,
)
from app.inference.dino.schema import DinoResult


class DinoService:
    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id
        self._detector: DinoDetector | None = None

    def detector(self) -> DinoDetector:
        if self._detector is None:
            self._detector = (
                DinoDetector(self._model_id) if self._model_id else DinoDetector()
            )
        return self._detector

    def detect(
        self,
        image: bytes,
        prompt: str,
        filename: str | None = None,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> DinoResult:
        if not image:
            raise ValueError("empty image")
        if not prompt or not prompt.strip():
            raise ValueError("empty prompt")
        try:
            rgb = bytes_to_rgb(image)
        except Exception as exc:
            raise ValueError("cannot decode image") from exc
        return self.detector().detect(
            rgb,
            prompt,
            filename=filename,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            max_size=max_size,
        )

    def release(self) -> None:
        detector = self._detector
        self._detector = None
        if detector is not None:
            detector.release()
