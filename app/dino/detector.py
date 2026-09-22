from __future__ import annotations

import inspect
import json
import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from app.dino.schema import DinoObject, DinoResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CACHE_DIR = PROJECT_ROOT / "models" / "hf"
SAVE_DIR = PROJECT_ROOT / "test" / "tmp"
DEFAULT_BOX_THRESHOLD = 0.4
DEFAULT_TEXT_THRESHOLD = 0.3
DEFAULT_MAX_SIZE = 800
JPEG_QUALITY = 85
COLORS = ("#22c55e", "#38bdf8", "#f59e0b", "#f43f5e", "#a78bfa")
log = logging.getLogger("roommind.dino")


def _is_jetson() -> bool:
    return Path("/etc/nv_tegra_release").is_file()


def _check_numpy() -> None:
    if not _is_jetson():
        return
    import numpy as np

    major = int(np.__version__.split(".", 1)[0])
    if major >= 2:
        raise RuntimeError(
            "Jetson 上的 PyTorch 需要 numpy 1.x，当前是 "
            f"{np.__version__}。请执行: pip install 'numpy>=1.23.5,<2'"
        )


def _device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def _release_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _allocated_mb() -> float | None:
    import torch

    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024**2), 1)
    mps = getattr(torch, "mps", None)
    if mps is not None and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return round(mps.current_allocated_memory() / (1024**2), 1)
    return None


def _reset_peak() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        torch.mps.synchronize()


def _peak_mb() -> float | None:
    import torch

    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    return _allocated_mb()


def _safe_stem(stem: str, fallback: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_")
    return safe or fallback


def bytes_to_rgb(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def fit_longest_edge(image: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / longest
    size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


def normalize_prompt(text: str) -> str:
    cleaned = (
        text.strip()
        .lower()
        .replace("。", ".")
        .replace("，", ".")
        .replace(",", ".")
    )
    parts = [item.strip() for item in cleaned.split(".") if item.strip()]
    if not parts:
        raise ValueError("empty prompt")
    return ". ".join(parts) + "."


def _as_xyxy(box: Any) -> list[float]:
    values = box.tolist() if hasattr(box, "tolist") else list(box)
    return [round(float(v), 1) for v in values]


def _scale_box(box: list[float], scale_x: float, scale_y: float) -> list[float]:
    x1, y1, x2, y2 = box
    return [
        round(x1 * scale_x, 1),
        round(y1 * scale_y, 1),
        round(x2 * scale_x, 1),
        round(y2 * scale_y, 1),
    ]


def _label_of(labels: Any, index: int) -> str:
    if labels is None:
        return ""
    value = labels[index]
    if hasattr(value, "item") and not isinstance(value, str):
        value = value.item()
    return str(value).strip()


def _score_of(score: Any) -> float:
    value = score.item() if hasattr(score, "item") else score
    return round(float(value), 4)


def render_boxes(image: Image.Image, objects: list[DinoObject]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    stroke = max(2, min(image.size) // 240)
    for index, obj in enumerate(objects):
        color = COLORS[index % len(COLORS)]
        x1, y1, x2, y2 = obj.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=stroke)
        caption = f"{obj.label} {obj.score:.2f}"
        text_bbox = draw.textbbox((x1, y1), caption, font=font)
        pad = 2
        background = [
            text_bbox[0] - pad,
            text_bbox[1] - pad,
            text_bbox[2] + pad,
            text_bbox[3] + pad,
        ]
        draw.rectangle(background, fill=color)
        draw.text((x1, y1), caption, fill="black", font=font)
    return annotated


def save_jpeg(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=JPEG_QUALITY)
    return path


def save_boxes_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _move_to_device(model: Any, device: str) -> Any:
    """逐个张量搬到 GPU。Jetson 是统一内存，整模 .to(cuda) 会再要一块约 1GB 的连续内存。"""
    if device == "cpu":
        return model
    import gc

    import torch

    for param in model.parameters():
        param.data = param.data.to(device=device)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(device=device)
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _post_process(
    processor: Any,
    outputs: Any,
    input_ids: Any,
    *,
    box_threshold: float,
    text_threshold: float,
    target_sizes: list[tuple[int, int]],
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    params = inspect.signature(processor.post_process_grounded_object_detection).parameters
    if "threshold" in params:
        kwargs["threshold"] = box_threshold
    elif "box_threshold" in params:
        kwargs["box_threshold"] = box_threshold
    else:
        kwargs["threshold"] = box_threshold
    return processor.post_process_grounded_object_detection(outputs, input_ids, **kwargs)


class DinoDetector:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        _check_numpy()
        import torch
        from transformers import AutoProcessor

        self.model_id = model_id
        self.device = _device_name()
        self.dtype = torch.float32
        self.model = None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR))
        log.info(
            "loading Grounding DINO model=%s device=%s dtype=%s cache=%s",
            model_id,
            self.device,
            str(self.dtype).replace("torch.", ""),
            CACHE_DIR,
        )
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=str(CACHE_DIR),
                local_files_only=True,
            )
        except OSError as exc:
            raise RuntimeError(
                "本地没有完整的 Grounding DINO 权重。在项目根目录执行:\n"
                "  huggingface-cli download IDEA-Research/grounding-dino-tiny "
                "--cache-dir \"$PWD/models/hf\""
            ) from exc
        try:
            self.model = self._load_model(self.dtype)
        except RuntimeError as exc:
            if not _is_oom(exc) or self.device != "cuda":
                raise
            log.warning("float32 加载显存不足，改用 float16")
            _release_cuda()
            self.dtype = torch.float16
            try:
                self.model = self._load_model(self.dtype)
            except RuntimeError as half_exc:
                if not _is_oom(half_exc):
                    raise
                raise RuntimeError(
                    "加载 Grounding DINO 时显存不足。Orin Nano 8GB 是统一内存，"
                    "请先停掉占内存的模型（ollama ps；ollama stop qwen3-vl:2b-instruct），"
                    "再用 free -h 确认至少有 2GB 可用。"
                ) from half_exc
        self.model.eval()
        self.vram_model_mb = _allocated_mb()
        log.info(
            "Grounding DINO ready cuda=%s vram_model_mb=%s",
            torch.cuda.is_available(),
            self.vram_model_mb,
        )

    def _load_model(self, dtype: Any) -> Any:
        from transformers import AutoModelForZeroShotObjectDetection

        kwargs = {
            "cache_dir": str(CACHE_DIR),
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "dtype": dtype,
        }
        try:
            model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id, **kwargs)
        except TypeError:
            kwargs.pop("dtype")
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                **kwargs,
            )
        except OSError as exc:
            raise RuntimeError(
                "本地没有完整的 Grounding DINO 权重。在项目根目录执行:\n"
                "  huggingface-cli download IDEA-Research/grounding-dino-tiny "
                "--cache-dir \"$PWD/models/hf\""
            ) from exc
        return _move_to_device(model, self.device)

    def _use_float16(self) -> None:
        log.warning("float32 推理显存不足，改用 float16")
        self.model = None
        _release_cuda()
        self.dtype = torch.float16
        self.model = self._load_model(self.dtype)
        self.model.eval()
        self.vram_model_mb = _allocated_mb()

    def _forward(self, inputs: Any) -> Any:
        import torch

        with torch.inference_mode():
            if self.device == "cuda" and self.dtype == torch.float16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    return self.model(**inputs)
            return self.model(**inputs)

    def detect(
        self,
        image: Image.Image,
        prompt: str,
        *,
        filename: str | None = None,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> DinoResult:
        if not 0 <= box_threshold <= 1:
            raise ValueError("box_threshold must be between 0 and 1")
        if not 0 <= text_threshold <= 1:
            raise ValueError("text_threshold must be between 0 and 1")

        text = normalize_prompt(prompt)
        work = fit_longest_edge(image, max_size)
        inputs = self.processor(images=work, text=text, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)

        _reset_peak()
        _synchronize()
        started = time.perf_counter()
        try:
            outputs = self._forward(inputs)
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            if self.dtype == torch.float16 or self.device != "cuda":
                raise RuntimeError(
                    "Grounding DINO 显存不足。请把 max_size 降到 640 或 512。"
                ) from exc
            self._use_float16()
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
            _reset_peak()
            started = time.perf_counter()
            try:
                outputs = self._forward(inputs)
            except RuntimeError as half_exc:
                if not _is_oom(half_exc):
                    raise
                raise RuntimeError(
                    "Grounding DINO 显存不足。请把 max_size 降到 640 或 512。"
                ) from half_exc
        _synchronize()
        inference_ms = round((time.perf_counter() - started) * 1000, 1)

        results = _post_process(
            self.processor,
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[work.size[::-1]],
        )
        raw = results[0] if results else {}
        boxes = raw.get("boxes", [])
        scores = raw.get("scores", [])
        labels = raw.get("text_labels", raw.get("labels"))
        scale_x = image.width / work.width
        scale_y = image.height / work.height
        objects: list[DinoObject] = []
        for index, box in enumerate(boxes):
            xyxy = _as_xyxy(box)
            if work.size != image.size:
                xyxy = _scale_box(xyxy, scale_x, scale_y)
            objects.append(
                DinoObject(
                    label=_label_of(labels, index),
                    score=_score_of(scores[index]) if index < len(scores) else 0.0,
                    bbox=xyxy,
                )
            )

        stem = _safe_stem(Path(filename or "dino").stem, "dino")
        boxes_path = SAVE_DIR / f"{stem}_boxes.json"
        annotated_path = SAVE_DIR / f"{stem}_annotated.jpg"
        save_boxes_json(
            {
                "prompt": text,
                "image_size": [image.width, image.height],
                "objects": [item.model_dump() for item in objects],
            },
            boxes_path,
        )
        save_jpeg(render_boxes(image, objects), annotated_path)
        log.info(
            "dino detections=%d inference_ms=%.1f vram_peak_mb=%s annotated=%s",
            len(objects),
            inference_ms,
            _peak_mb(),
            annotated_path,
        )
        return DinoResult(
            detection_count=len(objects),
            inference_ms=inference_ms,
            device=self.device,
            dtype=str(self.dtype).replace("torch.", ""),
            prompt=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            vram_model_mb=self.vram_model_mb,
            vram_peak_mb=_peak_mb(),
            objects=objects,
            boxes_path=str(boxes_path),
            annotated_path=str(annotated_path),
        )
