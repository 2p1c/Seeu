from __future__ import annotations

from PIL import Image

from app.inference.dinov3.encoder import Dinov3Encoder
from app.inference.trt.budget import claim_gpu


class Dinov3Service:
    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id
        self._encoder: Dinov3Encoder | None = None

    def encoder(self) -> Dinov3Encoder:
        if self._encoder is None:
            self._encoder = Dinov3Encoder(self._model_id)
        return self._encoder

    def embed(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        claim_gpu("dinov3", self.release)
        return self.encoder().embed(images)

    def release(self) -> None:
        encoder = self._encoder
        self._encoder = None
        if encoder is not None:
            encoder.release()
