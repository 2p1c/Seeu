import asyncio
import logging

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from pydantic import TypeAdapter

from app.vlm import OllamaQwen3VLAdapter, VLMResult, VLMService
from app.vlm.schema import HouseholdObject
from app.yolo import DetectionResult, YOLOService

app = FastAPI(title="RoomMind")
vlm_service = VLMService(OllamaQwen3VLAdapter())
yolo_service = YOLOService()
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


@app.post("/api/yolo/detect", response_model=DetectionResult)
async def detect(image: UploadFile = File(...)) -> DetectionResult:
    data = await image.read()
    log.info(
        "POST /api/yolo/detect filename=%s bytes=%d",
        image.filename,
        len(data),
    )
    try:
        result = await asyncio.to_thread(yolo_service.detect_image, data, image.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.info(
        "yolo result %s",
        result.model_dump_json(by_alias=True, exclude={"annotated_image"}),
    )
    return result
