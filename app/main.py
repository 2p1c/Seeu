import asyncio
import logging

from fastapi import FastAPI, File, Form, UploadFile

from app.vlm import OllamaQwen3VLAdapter, VLMResult, VLMService

app = FastAPI(title="RoomMind VLM")
vlm_service = VLMService(OllamaQwen3VLAdapter())
log = logging.getLogger("uvicorn.error")


@app.post("/api/vlm/analyze", response_model=VLMResult)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(...),
) -> VLMResult:
    data = await image.read()
    log.info(
        "POST /api/vlm/analyze filename=%s bytes=%d",
        image.filename,
        len(data),
    )
    return await asyncio.to_thread(vlm_service.analyze, data, prompt)
