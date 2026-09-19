import asyncio
import logging

from fastapi import FastAPI, File, Form, UploadFile

from pydantic import TypeAdapter

from app.vlm import OllamaQwen3VLAdapter, VLMResult, VLMService
from app.vlm.schema import HouseholdObject

app = FastAPI(title="RoomMind VLM")
vlm_service = VLMService(OllamaQwen3VLAdapter())
log = logging.getLogger("uvicorn.error")
_objects_adapter = TypeAdapter(list[HouseholdObject])


def _parse_objects(raw: str) -> list[HouseholdObject]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return _objects_adapter.validate_python(names)


@app.post("/api/vlm/analyze", response_model=VLMResult)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    time: str = Form(...),
    location: str = Form(...),
    objects: str = Form(...),
) -> VLMResult:
    data = await image.read()
    parsed_objects = _parse_objects(objects)
    log.info(
        "POST /api/vlm/analyze filename=%s bytes=%d time=%s location=%s objects=%s",
        image.filename,
        len(data),
        time,
        location,
        parsed_objects,
    )
    return await asyncio.to_thread(
        vlm_service.analyze,
        data,
        prompt,
        time,
        location,
        parsed_objects,
    )
