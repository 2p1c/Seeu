import asyncio
import logging
import re
import threading
import time
from pathlib import Path

import psycopg
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from app import object_memory
from app.camera.capture import camera_props, iter_frames, list_video_devices, open_capture, parse_source
from app.inference.dino import DinoResult, DinoService
from app.inference.dino.detector import (
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_MAX_SIZE as DINO_DEFAULT_MAX_SIZE,
    DEFAULT_TEXT_THRESHOLD,
)
from app.inference.sam import SAMResult, SAMService
from app.inference.sam.generator import (
    DEFAULT_MAX_SIZE,
    DEFAULT_POINTS_PER_BATCH,
    DEFAULT_POINTS_PER_CROP,
)
from app.inference.dinov3 import Dinov3CompareResult, Dinov3Result, Dinov3Service
from app.inference.image import open_rgb
from app.inference.scene import SceneState, SceneService
from app.inference.scene.service import latest_saved_scene
from app.inference.scene.progress import snapshot as scene_progress_snapshot
from app.inference.scene.select import DEFAULT_MAX_OBJECTS, DEFAULT_MIN_AREA_RATIO
from app.inference.siglip import SiglipService
from app.inference.trt.budget import drop_claims
from app.inference.trt.lease import GpuBusy, GpuLease
from app.inference.trt.plan import deployment_status
from app.inference.vlm import OllamaQwen3VLAdapter, VLMResult, VLMService
from app.inference.vlm.adapter import unload_model

app = FastAPI(title="RoomMind")
vlm_adapter = OllamaQwen3VLAdapter()
vlm_service = VLMService(vlm_adapter)
sam_service = SAMService()
dino_service = DinoService()
siglip_service = SiglipService()
dinov3_service = Dinov3Service()
scene_service = SceneService(
    sam_service,
    siglip_service,
    dinov3_service,
    vlm_adapter,
    also_release=(dino_service,),
)
log = logging.getLogger("uvicorn.error")
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "tests" / "tmp"
_ARTIFACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_camera_lock = threading.Lock()
_inference_lock = threading.Lock()
_camera_info: dict | None = None


def _locked(fn, *args):
    with GpuLease("perception", timeout_s=5):
        with _inference_lock:
            return fn(*args)


@app.exception_handler(GpuBusy)
async def gpu_busy_handler(_request, exc: GpuBusy):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(psycopg.OperationalError)
async def database_down_handler(_request, exc: psycopg.OperationalError):
    log.error("database unavailable: %s", exc)
    return JSONResponse(
        status_code=503,
        content={"detail": "物体记忆库连不上。检查 roomind-db 容器和 ROOMIND_DATABASE_URL。"},
    )


@app.get("/api/deploy")
def deploy() -> dict:
    return deployment_status()


@app.post("/api/deploy/release")
def deploy_release() -> dict:
    with _inference_lock:
        drop_claims()
        for service in (sam_service, dino_service, siglip_service, dinov3_service):
            service.release()
        unload_model()
    log.info("released loaded models")
    return {"released": True}


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
        _locked,
        vlm_service.analyze,
        data,
        prompt,
        time,
        location,
        parsed_objects,
    )


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
            _locked,
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
            _locked,
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


@app.post("/api/dinov3/embed", response_model=Dinov3Result)
async def dinov3_embed(image: UploadFile = File(...)) -> Dinov3Result:
    data = await image.read()
    log.info("POST /api/dinov3/embed filename=%s bytes=%d", image.filename, len(data))

    def embed() -> Dinov3Result:
        if not data:
            raise ValueError("empty image")
        try:
            rgb = open_rgb(data)
        except Exception as exc:
            raise ValueError("cannot decode image") from exc
        vectors = dinov3_service.embed([rgb])
        vector = vectors[0]
        return Dinov3Result(embedding=vector, dim=len(vector))

    try:
        result = await asyncio.to_thread(_locked, embed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    log.info("dinov3 dim=%d", result.dim)
    return result


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        raise ValueError("向量长度为 0")
    return dot / (left_norm * right_norm)


@app.post("/api/dinov3/compare", response_model=Dinov3CompareResult)
async def dinov3_compare(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
) -> Dinov3CompareResult:
    data_a = await image_a.read()
    data_b = await image_b.read()
    log.info(
        "POST /api/dinov3/compare a=%s bytes=%d b=%s bytes=%d",
        image_a.filename,
        len(data_a),
        image_b.filename,
        len(data_b),
    )

    def compare() -> Dinov3CompareResult:
        images = []
        for data in (data_a, data_b):
            if not data:
                raise ValueError("empty image")
            try:
                images.append(open_rgb(data))
            except Exception as exc:
                raise ValueError("cannot decode image") from exc
        vectors = dinov3_service.embed(images)
        score = _cosine(vectors[0], vectors[1])
        return Dinov3CompareResult(
            dim=len(vectors[0]),
            cosine_similarity=round(score, 4),
            embedding_a=vectors[0],
            embedding_b=vectors[1],
        )

    try:
        result = await asyncio.to_thread(_locked, compare)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    log.info("dinov3 cosine=%.4f dim=%d", result.cosine_similarity, result.dim)
    return result


@app.get("/api/scene/progress")
def scene_progress() -> dict:
    return scene_progress_snapshot()


@app.get("/api/scene/latest")
def scene_latest() -> dict:
    saved = latest_saved_scene()
    if saved is None:
        raise HTTPException(status_code=404, detail="还没有完成过的处理结果")
    return saved


@app.post("/api/scene", response_model=SceneState)
async def perceive_scene(
    image: UploadFile = File(...),
    time: str = Form(""),
    points_per_batch: int = Query(
        DEFAULT_POINTS_PER_BATCH,
        ge=1,
        description="SAM 每批点数，越小越省显存",
    ),
    points_per_crop: int = Query(
        DEFAULT_POINTS_PER_CROP,
        ge=1,
        description="SAM 每边采样点数",
    ),
    max_size: int = Query(
        DEFAULT_MAX_SIZE,
        ge=256,
        description="SAM 最长边上限",
    ),
    max_objects: int = Query(
        DEFAULT_MAX_OBJECTS,
        ge=1,
        le=30,
        description="送去分类、编码和描述的物体上限，按面积从大到小保留",
    ),
    min_area_ratio: float = Query(
        DEFAULT_MIN_AREA_RATIO,
        ge=0,
        le=1,
        description="小于画面面积这个比例的框会被丢掉",
    ),
) -> SceneState:
    data = await image.read()
    log.info(
        "POST /api/scene filename=%s bytes=%d points_per_batch=%d points_per_crop=%d max_size=%d max_objects=%d",
        image.filename,
        len(data),
        points_per_batch,
        points_per_crop,
        max_size,
        max_objects,
    )
    try:
        result = await asyncio.to_thread(
            _locked,
            scene_service.perceive,
            data,
            image.filename,
            time,
            points_per_batch,
            points_per_crop,
            max_size,
            max_objects,
            min_area_ratio,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    frame_id = await asyncio.to_thread(object_memory.save_scene, result)
    log.info(
        "scene result frame_id=%d %s",
        frame_id,
        result.model_dump_json(by_alias=True, exclude={"objects"}),
    )
    return result


@app.get("/api/memory/frames")
def memory_frames() -> list[dict]:
    return object_memory.list_frames()


@app.get("/api/memory/frames/{frame_id}")
def memory_frame(frame_id: int) -> dict:
    frame = object_memory.frame_detail(frame_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="没有这个画面")
    return frame


@app.get("/api/memory/latest")
def memory_latest() -> dict:
    frame = object_memory.latest_frame()
    if frame is None:
        raise HTTPException(status_code=404, detail="还没有记录过任何画面，先调用 POST /api/scene")
    return frame


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
