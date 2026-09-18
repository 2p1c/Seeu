from collections.abc import AsyncIterator

from app.vlm.adapter import OllamaQwen3VLAdapter
from app.vlm.schema import VLMResult


class VLMService:
    def __init__(self, adapter: OllamaQwen3VLAdapter) -> None:
        self._adapter = adapter

    def analyze(
        self,
        image: bytes,
        prompt: str,
        event_time: str,
        location: str,
    ) -> VLMResult:
        observation = self._adapter.analyze(image, prompt, location)
        return VLMResult(time=event_time, **observation.model_dump())

    async def analyze_video_stream(
        self,
        frames: AsyncIterator[bytes],
        prompt: str,
        event_time: str,
        location: str,
    ) -> AsyncIterator[VLMResult]:
        """Reserved: continuously read a video stream. Not started in this phase."""
        raise NotImplementedError
