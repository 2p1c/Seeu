import asyncio
import logging
import os

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile

from pydantic import TypeAdapter

from app.sam import SAMResult, SAMService
from app.sam.generator import (
    DEFAULT_MAX_SIZE,
    DEFAULT_POINTS_PER_BATCH,
    DEFAULT_POINTS_PER_CROP,
)
from app.vlm import OllamaQwen3VLAdapter, VLMResult, VLMService
from app.vlm.schema import HouseholdObject
from app.yolo import DetectionResult, YOLOService
from app.yolo.detector import DEFAULT_IMGSZ, DEFAULT_MODEL

app = FastAPI(title="RoomMind")
vlm_service = VLMService(OllamaQwen3VLAdapter())
yolo_service = YOLOService(os.environ.get("YOLO_MODEL", DEFAULT_MODEL))
sam_service = SAMService()
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
async def detect(
    image: UploadFile = File(...),
    imgsz: int = Query(DEFAULT_IMGSZ, ge=32, description="YOLO LetterBox 输入边长"),
) -> DetectionResult:
    data = await image.read()
    log.info(
        "POST /api/yolo/detect filename=%s bytes=%d imgsz=%d",
        image.filename,
        len(data),
        imgsz,
    )
    try:
        result = await asyncio.to_thread(
            yolo_service.detect_image,
            data,
            image.filename,
            imgsz,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    log.info(
        "yolo result %s",
        result.model_dump_json(by_alias=True, exclude={"annotated_image"}),
    )
    return result


@app.post("/api/sam/segment", response_model=SAMResult)
async def segment(
    image: UploadFile = File(...),
    points_per_batch: int = Query(
        DEFAULT_POINTS_PER_BATCH,
        ge=1,
        description="每批点数，越小越省显存，8GB 板建议 8 或 4",
    ),
    points_per_crop: int = Query(
        DEFAULT_POINTS_PER_CROP,
        ge=1,
        description="每边采样点数，16 表示 256 个点；默认 SAM 是 32",
    ),
    max_size: int = Query(
        DEFAULT_MAX_SIZE,
        ge=256,
        description="最长边上限，超出则缩小后再 AMG",
    ),
) -> SAMResult:
    data = await image.read()
    log.info(
        "POST /api/sam/segment filename=%s bytes=%d points_per_batch=%d points_per_crop=%d max_size=%d",
        image.filename,
        len(data),
        points_per_batch,
        points_per_crop,
        max_size,
    )
    try:
        result = await asyncio.to_thread(
            sam_service.segment,
            data,
            image.filename,
            points_per_batch,
            points_per_crop,
            max_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    log.info("sam result %s", result.model_dump_json())
    return result
