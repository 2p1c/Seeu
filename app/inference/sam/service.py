from __future__ import annotations

from PIL import Image

from app.inference.trt.budget import claim_gpu
from app.inference.sam.generator import (
    DEFAULT_MAX_SIZE,
    DEFAULT_POINTS_PER_BATCH,
    DEFAULT_POINTS_PER_CROP,
    SAMGenerator,
    bytes_to_rgb,
)
from app.inference.sam.masks import SegmentInstance
from app.inference.sam.schema import SAMResult


class SAMService:
    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id
        self._generator: SAMGenerator | None = None

    def generator(self) -> SAMGenerator:
        if self._generator is None:
            self._generator = SAMGenerator(self._model_id)
        return self._generator

    def segment(
        self,
        image: bytes,
        filename: str | None = None,
        points_per_batch: int = DEFAULT_POINTS_PER_BATCH,
        points_per_crop: int = DEFAULT_POINTS_PER_CROP,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> SAMResult:
        if not image:
            raise ValueError("empty image")
        try:
            rgb = bytes_to_rgb(image)
        except Exception as exc:
            raise ValueError("cannot decode image") from exc
        claim_gpu("sam", self.release)
        return self.generator().generate(
            rgb,
            filename=filename,
            points_per_batch=points_per_batch,
            points_per_crop=points_per_crop,
            max_size=max_size,
        )

    def instances(
        self,
        image: bytes,
        filename: str | None = None,
        points_per_batch: int = DEFAULT_POINTS_PER_BATCH,
        points_per_crop: int = DEFAULT_POINTS_PER_CROP,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> tuple[Image.Image, list[SegmentInstance]]:
        if not image:
            raise ValueError("empty image")
        try:
            rgb = bytes_to_rgb(image)
        except Exception as exc:
            raise ValueError("cannot decode image") from exc
        claim_gpu("sam", self.release)
        return self.generator().instances(
            rgb,
            filename=filename,
            points_per_batch=points_per_batch,
            points_per_crop=points_per_crop,
            max_size=max_size,
        )

    def release(self) -> None:
        generator = self._generator
        self._generator = None
        if generator is not None:
            generator.release()
