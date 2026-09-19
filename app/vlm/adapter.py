import logging
import threading
import time
from io import BytesIO

from httpx import Timeout
from ollama import Client
from PIL import Image

from app.vlm.schema import VLMObservation

MODEL_NAME = "qwen3-vl:2b-instruct"
MAX_IMAGE_SIDE = 640
NUM_CTX = 4096
log = logging.getLogger("uvicorn.error")

_client = Client(timeout=Timeout(300.0, connect=5.0))
_ps_client = Client(timeout=Timeout(5.0, connect=2.0))


def _mb(n: object) -> str:
    try:
        return f"{int(n) / 1024 / 1024:.0f}MB"
    except (TypeError, ValueError):
        return "?"


def _running_model() -> object | None:
    for model in _ps_client.ps().models:
        name = getattr(model, "model", None) or getattr(model, "name", "")
        if MODEL_NAME in str(name):
            return model
    return None


def _log_load_progress(stop: threading.Event) -> None:
    started = time.perf_counter()
    while not stop.is_set():
        elapsed = time.perf_counter() - started
        try:
            model = _running_model()
        except Exception:
            model = None
        if model is None:
            log.info("loading model %s ... waited %.0fs", MODEL_NAME, elapsed)
        else:
            log.info(
                "model loaded: %s size=%s vram=%s after %.0fs",
                MODEL_NAME,
                _mb(getattr(model, "size", 0)),
                _mb(getattr(model, "size_vram", 0)),
                elapsed,
            )
            return
        if stop.wait(2.0):
            return


def _prepare_image(image: bytes) -> bytes:
    img = Image.open(BytesIO(image)).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class OllamaQwen3VLAdapter:
    def analyze(self, image: bytes, prompt: str, location: str) -> VLMObservation:
        running = _running_model()
        stop = threading.Event()
        poller: threading.Thread | None = None
        if running is None:
            log.info("model not in memory, loading %s", MODEL_NAME)
            poller = threading.Thread(target=_log_load_progress, args=(stop,), daemon=True)
            poller.start()
        else:
            log.info(
                "model already loaded: %s size=%s vram=%s",
                MODEL_NAME,
                _mb(getattr(running, "size", 0)),
                _mb(getattr(running, "size_vram", 0)),
            )

        image = _prepare_image(image)
        log.info("ollama request start model=%s image_bytes=%d", MODEL_NAME, len(image))
        started = time.perf_counter()
        try:
            response = _client.chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "根据图片和用户提示作答。"
                            "objects 填看到的日常用品类别；"
                            "location 填位置描述；"
                            "description 只描述物体本身。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n参考位置：{location}",
                        "images": [image],
                    },
                ],
                format=VLMObservation.model_json_schema(),
                options={"num_ctx": NUM_CTX, "num_predict": 256},
            )
        finally:
            stop.set()
            if poller is not None:
                poller.join(timeout=1)

        log.info("ollama request done in %.1fs", time.perf_counter() - started)
        return VLMObservation.model_validate_json(response.message.content)
