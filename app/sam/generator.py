from __future__ import annotations

import logging
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from app.sam.schema import SAMResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "facebook/sam2.1-hiera-tiny"
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "sam"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
SAVE_DIR = PROJECT_ROOT / "test" / "tmp"
DEFAULT_POINTS_PER_BATCH = 8
DEFAULT_POINTS_PER_CROP = 16
DEFAULT_MAX_SIZE = 1024
JPEG_QUALITY = 85
log = logging.getLogger("roommind.sam")


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


def _pipeline_device() -> int | str:
    import torch

    if torch.cuda.is_available():
        return 0
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return -1


def _dtype_for(device: int | str) -> Any:
    import torch

    if device == -1:
        return torch.float32
    return torch.float16


def _device_name(device: int | str) -> str:
    if device == 0:
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


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
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "synchronize"):
        mps.synchronize()


def _peak_mb() -> float | None:
    import torch

    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024**2), 1)
    return _allocated_mb()


def _safe_stem(stem: str, fallback: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem).strip("_")
    return safe or fallback


def _configure_hf_endpoint() -> str:
    endpoint = os.environ.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")
    os.environ["HF_ENDPOINT"] = endpoint
    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.ENDPOINT = endpoint
        if hasattr(constants, "HF_HUB_OFFLINE"):
            constants.HF_HUB_OFFLINE = False
    return endpoint


def _local_model_ready(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return any(path.glob("*.safetensors")) or any(path.glob("*.bin"))


def _download_error(model_id: str, dest: Path) -> RuntimeError:
    endpoint = os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    return RuntimeError(
        f"无法下载 {model_id} 到 {dest}。板上访问 huggingface.co 经常不通。\n"
        f"当前 HF_ENDPOINT={endpoint}\n"
        "可以先设镜像再重试:\n"
        f"  export HF_ENDPOINT={DEFAULT_HF_ENDPOINT}\n"
        "或在能联网的机器上下载后拷到板上:\n"
        f"  huggingface-cli download {MODEL_ID} --local-dir {LOCAL_MODEL_DIR}"
        f" --endpoint {DEFAULT_HF_ENDPOINT}"
    )


def _ensure_model(model_id: str) -> str:
    candidate = Path(model_id).expanduser()
    if _local_model_ready(candidate):
        return str(candidate.resolve())
    if _local_model_ready(LOCAL_MODEL_DIR):
        return str(LOCAL_MODEL_DIR)

    repo_id = model_id if "/" in model_id and not candidate.exists() else MODEL_ID
    endpoint = _configure_hf_endpoint()
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s -> %s via %s", repo_id, LOCAL_MODEL_DIR, endpoint)
    offline = os.environ.pop("HF_HUB_OFFLINE", None)
    transformers_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, local_dir=str(LOCAL_MODEL_DIR), local_files_only=False)
    except Exception as exc:
        raise _download_error(repo_id, LOCAL_MODEL_DIR) from exc
    finally:
        if offline is not None:
            os.environ["HF_HUB_OFFLINE"] = offline
        if transformers_offline is not None:
            os.environ["TRANSFORMERS_OFFLINE"] = transformers_offline
    if not _local_model_ready(LOCAL_MODEL_DIR):
        raise _download_error(repo_id, LOCAL_MODEL_DIR)
    return str(LOCAL_MODEL_DIR)


_configure_hf_endpoint()


def bytes_to_rgb(data: bytes) -> Image.Image:
    image = Image.open(BytesIO(data)).convert("RGB")
    return image


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


def _as_bool_mask(mask: Any, height: int, width: int) -> Any:
    import numpy as np

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array.squeeze()
    binary = array.astype(bool)
    if binary.shape == (height, width):
        return binary
    vis = (binary.astype(np.uint8) * 255)
    resized = Image.fromarray(vis, mode="L").resize(
        (width, height),
        Image.Resampling.NEAREST,
    )
    return np.array(resized) > 0


def render_overlay(image: Image.Image, masks: list[Any]) -> Image.Image:
    import numpy as np

    base = np.array(image, dtype=np.float32)
    height, width = base.shape[:2]
    overlay = base.copy()
    rng = np.random.default_rng(0)
    for mask in masks:
        color = rng.integers(32, 256, size=3).astype(np.float32)
        selected = _as_bool_mask(mask, height, width)
        overlay[selected] = overlay[selected] * 0.45 + color * 0.55
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def render_masks(image: Image.Image, masks: list[Any]) -> Image.Image:
    import numpy as np

    height, width = image.size[1], image.size[0]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    rng = np.random.default_rng(1)
    for mask in masks:
        color = rng.integers(32, 256, size=3, dtype=np.uint8)
        canvas[_as_bool_mask(mask, height, width)] = color
    return Image.fromarray(canvas)


def save_jpeg(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=JPEG_QUALITY)
    return path


def save_png(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return path


class SAMGenerator:
    def __init__(self, model_id: str | None = None) -> None:
        _check_numpy()
        model_id = model_id or os.environ.get("SAM_MODEL") or MODEL_ID
        local_model = _ensure_model(model_id)
        import torch
        from transformers import pipeline
        self.device = _pipeline_device()
        self.dtype = _dtype_for(self.device)
        log.info(
            "loading SAM model=%s path=%s device=%s dtype=%s",
            model_id,
            local_model,
            _device_name(self.device),
            str(self.dtype).replace("torch.", ""),
        )
        self.pipeline = pipeline(
            "mask-generation",
            model=local_model,
            device=self.device,
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        self.vram_model_mb = _allocated_mb()
        log.info(
            "SAM ready cuda=%s vram_model_mb=%s",
            torch.cuda.is_available(),
            self.vram_model_mb,
        )

    def generate(
        self,
        image: Image.Image,
        *,
        filename: str | None = None,
        points_per_batch: int = DEFAULT_POINTS_PER_BATCH,
        points_per_crop: int = DEFAULT_POINTS_PER_CROP,
        max_size: int = DEFAULT_MAX_SIZE,
    ) -> SAMResult:
        if points_per_batch < 1:
            raise ValueError("points_per_batch must be >= 1")
        if points_per_crop < 1:
            raise ValueError("points_per_crop must be >= 1")

        work = fit_longest_edge(image, max_size)
        _reset_peak()
        _synchronize()
        started = time.perf_counter()
        try:
            outputs = self.pipeline(
                work,
                points_per_batch=points_per_batch,
                points_per_crop=points_per_crop,
                crops_n_layers=0,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                raise RuntimeError(
                    "SAM AMG 显存不足。请把 points_per_batch 降到 4 或 1，"
                    "或把 max_size 降到 768。"
                ) from exc
            raise
        _synchronize()
        inference_ms = round((time.perf_counter() - started) * 1000, 1)

        masks = list(outputs.get("masks") or [])
        stem = _safe_stem(Path(filename or "sam").stem, "sam")
        overlay_path = SAVE_DIR / f"{stem}_overlay.jpg"
        masks_path = SAVE_DIR / f"{stem}_masks.png"
        save_jpeg(render_overlay(work, masks), overlay_path)
        save_png(render_masks(work, masks), masks_path)
        log.info(
            "sam masks=%d inference_ms=%.1f vram_peak_mb=%s overlay=%s",
            len(masks),
            inference_ms,
            _peak_mb(),
            overlay_path,
        )
        return SAMResult(
            mask_count=len(masks),
            inference_ms=inference_ms,
            device=_device_name(self.device),
            dtype=str(self.dtype).replace("torch.", ""),
            points_per_batch=points_per_batch,
            points_per_crop=points_per_crop,
            vram_model_mb=self.vram_model_mb,
            vram_peak_mb=_peak_mb(),
            overlay_path=str(overlay_path),
            masks_path=str(masks_path),
        )
