import asyncio

from fastapi import FastAPI, File, Form, UploadFile

from app.vlm import OllamaQwen3VLAdapter, VLMResult, VLMService

app = FastAPI(title="RoomMind VLM")
vlm_service = VLMService(OllamaQwen3VLAdapter())


@app.post("/api/vlm/analyze", response_model=VLMResult)
async def analyze(
    image: UploadFile = File(...),
    prompt: str = Form(...),
) -> VLMResult:
    data = await image.read()
    return await asyncio.to_thread(vlm_service.analyze, data, prompt)
