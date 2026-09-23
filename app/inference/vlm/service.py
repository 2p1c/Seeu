from collections.abc import AsyncIterator

from app.inference.vlm.adapter import OllamaQwen3VLAdapter
from app.inference.vlm.schema import VLMResult


class VLMService:
    def __init__(self, adapter: OllamaQwen3VLAdapter) -> None:
        self._adapter = adapter

    def analyze(
        self,
        image: bytes,
        prompt: str,
        event_time: str,
        location: str,
        objects: list[str],
    ) -> VLMResult:
        observation = self._adapter.analyze(image, prompt, location, list(objects))
        return VLMResult(
            time=event_time,
            objects=objects,
            location=location,
            description=observation.description,
        )

    async def analyze_video_stream(
        self,
        frames: AsyncIterator[bytes],
        prompt: str,
        event_time: str,
        location: str,
        objects: list[str],
    ) -> AsyncIterator[VLMResult]:
        """Reserved: continuously read a video stream. Not started in this phase."""
        raise NotImplementedError
