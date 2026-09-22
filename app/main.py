import asyncio
import logging
import os
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.camera.capture import camera_props, iter_frames, list_video_devices, open_capture, parse_source
from app.dino import DinoResult, DinoService
from app.dino.detector import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_MAX_SIZE as DINO_DEFAULT_MAX_SIZE,
    DEFAULT_TEXT_THRESHOLD,
)
from app.sam import SAMResult, SAMService
from app.sam.generator import (
    DEFAULT_MAX_SIZE,
    DEFAULT_POINTS_PER_BATCH,
    DEFAULT_POINTS_PER_CROP,
)
from app.vlm import OllamaQwen3VLAdapter, VLMResult, VLMService
from app.yolo import DetectionResult, YOLOService
from app.yolo.detector import DEFAULT_IMGSZ, DEFAULT_MODEL

app = FastAPI(title="RoomMind")
vlm_service = VLMService(OllamaQwen3VLAdapter())
yolo_service = YOLOService(os.environ.get("YOLO_MODEL", DEFAULT_MODEL))
sam_service = SAMService()
dino_service = DinoService()
log = logging.getLogger("uvicorn.error")
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "test" / "tmp"
_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_camera_lock = threading.Lock()
_camera_info: dict | None = None


def _parse_objects(raw: str) -> list[str]:
    names = [item.strip() for item in re.split(r"[,，、;；\n]+", raw) if item.strip()]
    if not names:
        raise ValueError("empty objects")
    return names


@app.post("/api/vlm/analyze", response_model=VLMResult)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    time: str = Form(...),
    location: str = Form(...),
    objects: str = Form(...),
) -> VLMResult:
    data = await image.read()
    try:
        parsed_objects = _parse_objects(objects)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="请填写要描述的物体") from exc
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


@app.post("/api/dino/detect", response_model=DinoResult)
async def dino_detect(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    box_threshold: float = Query(
        DEFAULT_BOX_THRESHOLD,
        ge=0,
        le=1,
        description="检测框置信度阈值，越大框越少",
    ),
    text_threshold: float = Query(
        DEFAULT_TEXT_THRESHOLD,
        ge=0,
        le=1,
        description="文本匹配阈值",
    ),
    max_size: int = Query(
        DINO_DEFAULT_MAX_SIZE,
        ge=256,
        description="最长边上限，超出则缩小后再推理；8GB 板建议 800 或 640",
    ),
) -> DinoResult:
    data = await image.read()
    log.info(
        "POST /api/dino/detect filename=%s bytes=%d prompt=%s box_threshold=%.2f text_threshold=%.2f max_size=%d",
        image.filename,
        len(data),
        prompt,
        box_threshold,
        text_threshold,
        max_size,
    )
    try:
        result = await asyncio.to_thread(
            dino_service.detect,
            data,
            prompt,
            image.filename,
            box_threshold,
            text_threshold,
            max_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    log.info(
        "dino result %s",
        result.model_dump_json(exclude={"objects"}),
    )
    return result


@app.get("/api/camera/devices")
def camera_devices() -> list[dict[str, str]]:
    return [{"source": source, "name": name} for source, name in list_video_devices()]


@app.get("/api/camera/info")
def camera_info() -> dict:
    if _camera_info is None:
        return {"open": False}
    return {"open": True, **_camera_info}


@app.get("/api/camera/stream")
def camera_stream(
    source: str = Query("0"),
    width: int = Query(1920, ge=160, le=3840),
    height: int = Query(1080, ge=120, le=2160),
    max_fps: float = Query(10, ge=1, le=30),
) -> StreamingResponse:
    global _camera_info
    if not _camera_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="摄像头正在使用")
    try:
        cap = open_capture(parse_source(source), width=width, height=height)
    except Exception as exc:
        _camera_lock.release()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    props = camera_props(cap)
    _camera_info = {
        "open": True,
        "source": source,
        "requested_width": width,
        "requested_height": height,
        "max_fps": max_fps,
        "measured_fps": 0.0,
        **props,
    }

    def frames():
        global _camera_info
        count = 0
        window = time.monotonic()
        try:
            import cv2

            for frame in iter_frames(cap, max_fps=max_fps):
                ok, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                )
                if not ok:
                    continue
                payload = buf.tobytes()
                count += 1
                now = time.monotonic()
                if _camera_info is not None and now - window >= 1:
                    _camera_info["measured_fps"] = round(count / (now - window), 1)
                    count = 0
                    window = now
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                )
        finally:
            cap.release()
            _camera_info = None
            _camera_lock.release()
            log.info("camera stream closed source=%s", source)

    log.info("camera stream source=%s %sx%s max_fps=%s", source, width, height, max_fps)
    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/artifacts/{name}")
def artifact(name: str) -> FileResponse:
    if not _ARTIFACT_NAME.fullmatch(name):
        raise HTTPException(status_code=400, detail="invalid artifact name")
    path = (ARTIFACT_DIR / name).resolve()
    if path.parent != ARTIFACT_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    media = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".json": "application/json",
    }.get(path.suffix.lower())
    if media is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path, media_type=media)
