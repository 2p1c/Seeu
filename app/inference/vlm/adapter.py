import logging
import os
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

from httpx import Timeout
import httpx
from PIL import Image, ImageDraw

from app.inference.trt.budget import claim_gpu
from app.inference.vlm.schema import ObjectDescription, VLMObservation

MODEL_NAME = "qwen3-vl:2b-instruct"
MAX_IMAGE_SIDE = 640
# 一次请求最多两张 640px 图（各约 400 token）加提示词，4K 足够。KV cache 随它线性增长，
# 8GB Jetson 上 25.6K 会让 Ollama 申请不到 KV cache。
NUM_CTX = 4096
NUM_PREDICT = 1024
log = logging.getLogger("uvicorn.error")

_PROXY_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "SOCKS_PROXY",
    "socks_proxy",
)


def _ollama_client_cls():
    # ollama 在 import 时就会 Client()；本机 11434 不该走 SOCKS，否则缺 socksio 会直接崩。
    saved = {key: os.environ.pop(key, None) for key in _PROXY_KEYS}
    try:
        from ollama import Client
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
    return Client


Client = _ollama_client_cls()
_client = Client(timeout=Timeout(300.0, connect=5.0), trust_env=False)
_ps_client = Client(timeout=Timeout(5.0, connect=2.0), trust_env=False)


def _save_response(text: str) -> Path:
    directory = Path(__file__).resolve().parents[3] / "tests" / "tmp"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"vlm_{datetime.now().strftime('%H%M%S_%f')}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _mb(n: object) -> str:
    try:
        return f"{int(n) / 1024 / 1024:.0f}MB"
    except (TypeError, ValueError):
        return "?"


def ollama_memory_mb() -> tuple[float | None, float | None]:
    """返回 Ollama 里该模型的 (总占用 MB, 显存占用 MB)。"""
    try:
        model = _running_model()
    except Exception:
        return None, None
    if model is None:
        return None, None

    def as_mb(value: object) -> float | None:
        try:
            return round(int(value) / 1024 / 1024, 1)
        except (TypeError, ValueError):
            return None

    return as_mb(getattr(model, "size", 0)), as_mb(getattr(model, "size_vram", 0))


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
    from app.inference.image import open_rgb

    img = open_rgb(image)
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _jpeg_and_scale(image: Image.Image) -> tuple[bytes, float, float]:
    width, height = image.size
    resized = image.copy()
    resized.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=85)
    scale_x = resized.width / width if width else 1.0
    scale_y = resized.height / height if height else 1.0
    return buf.getvalue(), scale_x, scale_y


def unload_model() -> None:
    try:
        running = _running_model()
    except Exception:
        return
    if running is None:
        return
    log.info("unloading %s to free memory", MODEL_NAME)
    try:
        httpx.post(
            "http://127.0.0.1:11434/api/generate",
            json={"model": MODEL_NAME, "keep_alive": 0},
            timeout=30.0,
            trust_env=False,
        )
    except Exception as exc:
        log.warning("unload %s failed: %s", MODEL_NAME, exc)
        return
    deadline = time.perf_counter() + 15
    while time.perf_counter() < deadline:
        try:
            if _running_model() is None:
                log.info("unloaded %s", MODEL_NAME)
                return
        except Exception:
            return
        time.sleep(0.5)
    log.warning("%s still in memory after unload", MODEL_NAME)


class OllamaQwen3VLAdapter:
    def analyze(
        self,
        image: bytes,
        prompt: str,
        location: str,
        objects: list[str],
    ) -> VLMObservation:
        claim_gpu("vlm", unload_model, unload_ollama=False)
        image = _prepare_image(image)
        content = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "根据图片、已知物体和位置作答。"
                        "只描述这些物体本身，不要列举类别名。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n"
                        f"已知物体：{', '.join(objects)}\n"
                        f"位置：{location}"
                    ),
                    "images": [image],
                },
            ],
            VLMObservation.model_json_schema(),
            image_bytes=len(image),
        )
        return VLMObservation.model_validate_json(content)

    def describe_object(
        self,
        scene: Image.Image,
        crop: Image.Image,
        bbox: list[int],
    ) -> str:
        claim_gpu("vlm", unload_model, unload_ollama=False)
        marked = scene.copy()
        draw = ImageDraw.Draw(marked)
        x1, y1, x2, y2 = bbox
        width = max(2, min(marked.size) // 200)
        draw.rectangle((x1, y1, x2 + 1, y2 + 1), outline=(255, 48, 48), width=width)
        scene_bytes, scale_x, scale_y = _jpeg_and_scale(marked)
        crop_bytes, _, _ = _jpeg_and_scale(crop)
        shown_w = max(1, round(marked.width * scale_x))
        shown_h = max(1, round(marked.height * scale_y))
        scaled = [
            round(x1 * scale_x, 1),
            round(y1 * scale_y, 1),
            round(x2 * scale_x, 1),
            round(y2 * scale_y, 1),
        ]
        content = self._chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你在描述画面中的一件物品。"
                        "第一张图是完整画面，红框标出目标。第二张图是该物品的裁剪。"
                        "只描述这一件物品本身，以及它在画面中的位置。"
                        "回复最多100字左右，写完即止。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "请描述红框中的物品本身，以及它在画面里的位置。\n"
                        f"边界框 {scaled}，画面尺寸 {shown_w}x{shown_h}。"
                    ),
                    "images": [scene_bytes, crop_bytes],
                },
            ],
            ObjectDescription.model_json_schema(),
            image_bytes=len(scene_bytes) + len(crop_bytes),
        )
        return ObjectDescription.model_validate_json(content).description

    def _chat(self, messages: list[dict], schema: dict, image_bytes: int) -> str:
        try:
            running = _running_model()
        except Exception as exc:
            raise RuntimeError(
                "无法连接 Ollama。请先启动 ollama serve，并执行 ollama pull qwen3-vl:2b-instruct。"
            ) from exc
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

        log.info("ollama request start model=%s image_bytes=%d", MODEL_NAME, image_bytes)
        started = time.perf_counter()
        try:
            response = _client.chat(
                model=MODEL_NAME,
                messages=messages,
                format=schema,
                options={"num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
            )
        except Exception as exc:
            text = str(exc).lower()
            if "connection" in text or "11434" in text or "refused" in text:
                raise RuntimeError(
                    "无法连接 Ollama。请先启动 ollama serve，并执行 ollama pull qwen3-vl:2b-instruct。"
                ) from exc
            raise
        finally:
            stop.set()
            if poller is not None:
                poller.join(timeout=1)

        log.info("ollama request done in %.1fs", time.perf_counter() - started)
        text = response.message.content or ""
        saved = _save_response(text)
        log.info(
            "ollama response chars=%d saved=%s head=%s tail=%s",
            len(text),
            saved,
            text[:120].replace("\n", " "),
            text[-120:].replace("\n", " "),
        )
        return text
