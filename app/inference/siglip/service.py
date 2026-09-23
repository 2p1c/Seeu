from __future__ import annotations

from PIL import Image

from app.inference.siglip.classifier import TOP_K, SiglipClassifier
from app.inference.siglip.schema import ClassScore


class SiglipService:
    def __init__(self, model_id: str | None = None) -> None:
        self._model_id = model_id
        self._classifier: SiglipClassifier | None = None

    def classifier(self) -> SiglipClassifier:
        if self._classifier is None:
            self._classifier = SiglipClassifier(self._model_id)
        return self._classifier

    def classify(self, images: list[Image.Image], top_k: int = TOP_K) -> list[list[ClassScore]]:
        if not images:
            return []
        return self.classifier().classify(images, top_k=top_k)

    def release(self) -> None:
        classifier = self._classifier
        self._classifier = None
        if classifier is not None:
            classifier.release()
